use std::ffi::CString;
use std::io::{self, IsTerminal, Write};
use std::os::raw::c_char;
use std::process::ExitCode;
use std::ptr;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "spawnado",
    version,
    max_term_width = 80,
    about = "Benchmark posix_spawn(3C) performance, \
             optionally under thread contention",
    long_about = "Benchmarks posix_spawn(3C) by repeatedly spawning a child \
        process from one or more threads. Each iteration times only the \
        posix_spawn call itself; the child is then reaped with waitpid(2) \
        outside the measured window. \n\n\
        In sweep mode (--sweep) the benchmark is run once per requested \
        thread count and a comparison table is printed at the end so you \
        can see how posix_spawn scales with concurrency."
)]
struct Args {
    /// Number of worker threads spawning concurrently.
    #[arg(
        short = 'j',
        long,
        default_value_t = 1,
        value_name = "N",
        conflicts_with = "sweep"
    )]
    threads: usize,

    /// Sweep across a list of thread counts.
    ///
    /// Accepts comma-separated integers and inclusive ranges, e.g.
    /// "1,2,4,8", "1-8", or "1,2,4-8,16". Mutually exclusive with --threads.
    #[arg(long, value_name = "SPEC")]
    sweep: Option<String>,

    /// Measured spawns per thread (per run).
    #[arg(short = 'n', long, default_value_t = 1000, value_name = "N")]
    iterations: usize,

    /// Warmup spawns per thread (not measured).
    #[arg(short = 'w', long, default_value_t = 10, value_name = "N")]
    warmup: usize,

    /// Suppress ANSI colour in output.
    #[arg(long, alias = "no-color")]
    no_colour: bool,

    /// Emit machine-readable CSV instead of the human-readable report.
    ///
    /// One row per run with full statistics; suitable for piping into
    /// gnuplot, awk, or a spreadsheet. Implies --no-colour.
    #[arg(long)]
    csv: bool,

    /// Command to spawn (default: /usr/bin/true). Trailing arguments are
    /// passed verbatim.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
    command: Vec<String>,
}

fn parse_sweep(s: &str) -> Result<Vec<usize>, String> {
    let mut out: Vec<usize> = Vec::new();
    for raw in s.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        if let Some((low, high)) = token.split_once('-') {
            let low: usize = low
                .trim()
                .parse()
                .map_err(|_| format!("invalid number in range: {low:?}"))?;
            let high: usize = high
                .trim()
                .parse()
                .map_err(|_| format!("invalid number in range: {high:?}"))?;
            if low == 0 || high == 0 {
                return Err("thread counts must be >= 1".into());
            }
            if low > high {
                return Err(format!("range {low}-{high} is descending"));
            }
            out.extend(low..=high);
        } else {
            let n: usize = token
                .parse()
                .map_err(|_| format!("invalid thread count: {token:?}"))?;
            if n == 0 {
                return Err("thread counts must be >= 1".into());
            }
            out.push(n);
        }
    }
    if out.is_empty() {
        return Err("sweep specification is empty".into());
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Owns the CStrings backing argv and exposes a stable `*const *mut c_char`.
struct Argv {
    _strings: Vec<CString>,
    pointers: Vec<*mut c_char>,
}

// Safety: the inner pointers refer to heap-allocated CString data that we own
// and never mutate; multiple threads reading them concurrently is fine.
unsafe impl Send for Argv {}
unsafe impl Sync for Argv {}

impl Argv {
    fn new(binary: &str, args: &[String]) -> io::Result<Self> {
        let to_cstring = |s: &str| {
            CString::new(s).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("argument contains NUL byte: {e}"),
                )
            })
        };
        let mut strings = Vec::with_capacity(args.len() + 1);
        strings.push(to_cstring(binary)?);
        for a in args {
            strings.push(to_cstring(a)?);
        }
        let mut pointers: Vec<*mut c_char> =
            strings.iter().map(|s| s.as_ptr() as *mut c_char).collect();
        pointers.push(ptr::null_mut());
        Ok(Self { _strings: strings, pointers })
    }

    fn as_ptr(&self) -> *const *mut c_char {
        self.pointers.as_ptr()
    }
}

/// Spawn the configured command once, returning the wall-clock time the
/// posix_spawn call itself consumed. The child is reaped synchronously
/// after the measurement window.
fn spawn_one(path: &CString, argv: *const *mut c_char) -> io::Result<Duration> {
    let mut pid: libc::pid_t = 0;
    let envp: [*mut c_char; 1] = [ptr::null_mut()];

    let start = Instant::now();
    let rc = unsafe {
        libc::posix_spawn(
            &mut pid,
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            argv,
            envp.as_ptr() as *const *mut c_char,
        )
    };
    let elapsed = start.elapsed();

    if rc != 0 {
        return Err(io::Error::from_raw_os_error(rc));
    }

    let mut status: libc::c_int = 0;
    loop {
        let r = unsafe { libc::waitpid(pid, &mut status, 0) };
        if r == pid {
            break;
        }
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        return Err(err);
    }
    Ok(elapsed)
}

#[derive(Clone, Copy)]
struct Style {
    bold: &'static str,
    dim: &'static str,
    green: &'static str,
    cyan: &'static str,
    reset: &'static str,
}

impl Style {
    fn new(enable: bool) -> Self {
        if enable {
            Self {
                bold: "\x1b[1m",
                dim: "\x1b[2m",
                green: "\x1b[32;1m",
                cyan: "\x1b[36m",
                reset: "\x1b[0m",
            }
        } else {
            Self { bold: "", dim: "", green: "", cyan: "", reset: "" }
        }
    }
}

fn pick_unit(typical_ns: f64) -> (f64, &'static str) {
    if typical_ns < 1_000.0 {
        (1.0, "ns")
    } else if typical_ns < 1_000_000.0 {
        (1_000.0, "µs")
    } else if typical_ns < 1_000_000_000.0 {
        (1_000_000.0, "ms")
    } else {
        (1_000_000_000.0, "s")
    }
}

fn fmt_dur(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    let (div, label) = pick_unit(ns);
    format!("{:.2} {}", ns / div, label)
}

struct Stats {
    n: usize,
    mean_ns: f64,
    stddev_ns: f64,
    min_ns: f64,
    max_ns: f64,
    p50_ns: f64,
    p95_ns: f64,
    p99_ns: f64,
}

impl Stats {
    fn from_sorted(sorted: &[Duration]) -> Self {
        let n = sorted.len();
        debug_assert!(n > 0);
        let to_ns = |d: Duration| d.as_nanos() as f64;
        let sum: f64 = sorted.iter().copied().map(to_ns).sum();
        let mean = sum / n as f64;
        let var: f64 = sorted
            .iter()
            .copied()
            .map(|d| {
                let x = to_ns(d) - mean;
                x * x
            })
            .sum::<f64>()
            / n as f64;
        // Nearest-rank percentile, clamped so p=1.0 hits the last element.
        let pct = |p: f64| -> f64 {
            let idx = ((n as f64) * p).ceil() as usize;
            let idx = idx.saturating_sub(1).min(n - 1);
            to_ns(sorted[idx])
        };
        Self {
            n,
            mean_ns: mean,
            stddev_ns: var.sqrt(),
            min_ns: to_ns(*sorted.first().unwrap()),
            max_ns: to_ns(*sorted.last().unwrap()),
            p50_ns: pct(0.50),
            p95_ns: pct(0.95),
            p99_ns: pct(0.99),
        }
    }
}

struct RunResult {
    threads: usize,
    stats: Stats,
    wall: Duration,
}

impl RunResult {
    fn throughput(&self) -> f64 {
        self.stats.n as f64 / self.wall.as_secs_f64()
    }
}

fn run_bench(
    threads: usize,
    iterations: usize,
    warmup: usize,
    cpath: &Arc<CString>,
    argv: &Arc<Argv>,
) -> io::Result<RunResult> {
    let barrier = Arc::new(Barrier::new(threads));
    let total = iterations.saturating_mul(threads);

    let bench_start = Instant::now();
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let cpath = Arc::clone(cpath);
        let argv = Arc::clone(argv);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || -> io::Result<Vec<Duration>> {
            for _ in 0..warmup {
                spawn_one(&cpath, argv.as_ptr())?;
            }
            // All threads cross together so the measured window genuinely
            // overlaps across workers.
            barrier.wait();
            let mut samples = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                samples.push(spawn_one(&cpath, argv.as_ptr())?);
            }
            Ok(samples)
        }));
    }

    let mut all = Vec::with_capacity(total);
    for h in handles {
        let part = h
            .join()
            .map_err(|_| io::Error::other("worker thread panicked"))??;
        all.extend(part);
    }
    let wall = bench_start.elapsed();
    all.sort_unstable();
    Ok(RunResult { threads, stats: Stats::from_sorted(&all), wall })
}

fn print_run(result: &RunResult, style: Style) {
    let s = &result.stats;
    let (div, unit) = pick_unit(s.mean_ns);
    let to_unit = |ns: f64| ns / div;

    let Style { bold, dim, green, cyan, reset } = style;

    println!(
        "  {bold}Time (mean ± σ):{reset}     \
         {green}{:>9.2} {unit}{reset} ± {:>8.2} {unit}",
        to_unit(s.mean_ns),
        to_unit(s.stddev_ns),
    );
    println!(
        "  {dim}Range (min … max):{reset}   \
         {:>9.2} {unit} … {:>8.2} {unit}    {} runs",
        to_unit(s.min_ns),
        to_unit(s.max_ns),
        s.n,
    );
    println!(
        "  {dim}Percentiles:{reset}         \
         p50 {:.2} {unit}   p95 {:.2} {unit}   \
         p99 {:.2} {unit}",
        to_unit(s.p50_ns),
        to_unit(s.p95_ns),
        to_unit(s.p99_ns),
    );
    println!();
    println!(
        "  {bold}Throughput:{reset}         \
         {cyan}{:>9.0}{reset} spawns/sec   \
         ({} wall, {} thread{})",
        result.throughput(),
        fmt_dur(result.wall),
        result.threads,
        if result.threads == 1 { "" } else { "s" },
    );
}

fn print_sweep_summary(results: &[RunResult], style: Style) {
    if results.is_empty() {
        return;
    }
    // Pick a single time unit from the largest mean so the column lines up.
    let max_mean =
        results.iter().map(|r| r.stats.mean_ns).fold(0.0_f64, f64::max);
    let (div, unit) = pick_unit(max_mean);
    let to_unit = |ns: f64| ns / div;

    let baseline_throughput = results[0].throughput();
    let best = results
        .iter()
        .max_by(|a, b| a.throughput().partial_cmp(&b.throughput()).unwrap())
        .unwrap();

    let Style { bold, dim, green, cyan, reset } = style;

    println!();
    println!(
        "{bold}Summary{reset} {dim}(baseline = {} thread{}){reset}",
        results[0].threads,
        if results[0].threads == 1 { "" } else { "s" },
    );

    let header = format!(
        "  {:>7}  {:>12}  {:>12}  {:>12}  {:>12}  {:>14}  {:>9}",
        "Threads", "Mean", "StdDev", "Min", "Max", "Throughput", "Speedup",
    );
    println!("{bold}{header}{reset}");
    println!("  {}", "─".repeat(header.len().saturating_sub(2)));
    for r in results {
        let s = &r.stats;
        let speedup = r.throughput() / baseline_throughput;
        let mean_col = format!("{:.2} {}", to_unit(s.mean_ns), unit);
        let std_col = format!("{:.2} {}", to_unit(s.stddev_ns), unit);
        let min_col = format!("{:.2} {}", to_unit(s.min_ns), unit);
        let max_col = format!("{:.2} {}", to_unit(s.max_ns), unit);
        let tput_col = format!("{:.0}/s", r.throughput());
        let highlight = if std::ptr::eq(r, best) { green } else { "" };
        let highlight_reset = if std::ptr::eq(r, best) { reset } else { "" };
        println!(
            "  {highlight}{:>7}  {:>12}  {:>12}  {:>12}  \
             {:>12}  {:>14}  {:>8.2}×{highlight_reset}",
            r.threads, mean_col, std_col, min_col, max_col, tput_col, speedup,
        );
    }
    println!();
    println!(
        "  {bold}Best throughput:{reset} {cyan}{:.0} spawns/sec{reset} \
         at {} thread{}",
        best.throughput(),
        best.threads,
        if best.threads == 1 { "" } else { "s" },
    );
}

/// Round a positive value up to a "nice" axis maximum (1, 2, 5 * 10^k).
fn nice_axis_max(v: f64) -> f64 {
    if v <= 0.0 {
        return 1.0;
    }
    let mag = 10.0_f64.powf(v.log10().floor());
    let n = v / mag;
    let nice = if n <= 1.0 {
        1.0
    } else if n <= 2.0 {
        2.0
    } else if n <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * mag
}

fn print_throughput_plot(results: &[RunResult], style: Style) {
    if results.len() < 2 {
        return;
    }

    const HEIGHT: usize = 11;
    const COL: usize = 6;
    let width = results.len() * COL;

    let max_tp = results.iter().map(|r| r.throughput()).fold(0.0_f64, f64::max);
    let axis_max = nice_axis_max(max_tp);

    let mut grid: Vec<Vec<char>> = vec![vec![' '; width]; HEIGHT];
    for (idx, r) in results.iter().enumerate() {
        let col = idx * COL + 2;
        let frac = r.throughput() / axis_max;
        let row = ((1.0 - frac) * (HEIGHT - 1) as f64).round() as usize;
        let row = row.min(HEIGHT - 1);
        if col < width {
            grid[row][col] = '*';
        }
    }

    let Style { bold, dim, cyan, reset, .. } = style;

    println!();
    println!("{bold}Throughput vs threads (spawns/sec):{reset}");

    for (i, row_chars) in grid.iter().enumerate() {
        let show = i == 0 || i == HEIGHT - 1 || i == HEIGHT / 2;
        let y_val = axis_max * (1.0 - i as f64 / (HEIGHT - 1) as f64);
        let label = if show { format!("{y_val:>8.0}") } else { " ".repeat(8) };
        let row_str: String = row_chars.iter().collect();
        println!("  {dim}{label} |{reset}{cyan}{row_str}{reset}");
    }

    let axis: String = "-".repeat(width);
    println!("  {dim}{:>8} +{axis}{reset}", "");

    let mut labels = String::new();
    for r in results {
        labels.push_str(&format!("{:^6}", r.threads));
    }
    println!("  {:>8}  {dim}{labels}{reset}", "");
    println!("  {:>8}  {dim}threads{reset}", "");
}

fn print_csv(results: &[RunResult]) {
    println!(
        "threads,samples,mean_ns,stddev_ns,min_ns,max_ns,\
         p50_ns,p95_ns,p99_ns,throughput_per_sec,wall_secs"
    );
    for r in results {
        let s = &r.stats;
        println!(
            "{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6}",
            r.threads,
            s.n,
            s.mean_ns,
            s.stddev_ns,
            s.min_ns,
            s.max_ns,
            s.p50_ns,
            s.p95_ns,
            s.p99_ns,
            r.throughput(),
            r.wall.as_secs_f64(),
        );
    }
}

fn run() -> io::Result<()> {
    let args = Args::parse();

    if args.threads == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--threads must be >= 1",
        ));
    }
    if args.iterations == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--iterations must be >= 1",
        ));
    }

    let (binary, binary_args): (String, Vec<String>) =
        if args.command.is_empty() {
            ("/usr/bin/true".to_string(), Vec::new())
        } else {
            let mut it = args.command.into_iter();
            let head = it.next().unwrap();
            (head, it.collect())
        };

    let stdout_tty = io::stdout().is_terminal();
    let use_color = !args.no_colour && !args.csv && stdout_tty;
    let style = Style::new(use_color);
    let human = !args.csv;

    let cpath = CString::new(binary.as_str()).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidInput, e.to_string())
    })?;
    let argv = Argv::new(&binary, &binary_args)?;

    // Surface ENOENT / EACCES once up front, instead of from every worker.
    spawn_one(&cpath, argv.as_ptr()).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("initial spawn of {binary:?} failed: {e}"),
        )
    })?;

    let cpath = Arc::new(cpath);
    let argv = Arc::new(argv);

    let cmdline = if binary_args.is_empty() {
        binary.clone()
    } else {
        format!("{binary} {}", binary_args.join(" "))
    };
    let thread_counts: Vec<usize> = match args.sweep.as_deref() {
        Some(spec) => parse_sweep(spec).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("--sweep: {e}"))
        })?,
        None => vec![args.threads],
    };

    if human {
        println!("{}Target{}: {}", style.bold, style.reset, cmdline);
        if thread_counts.len() > 1 {
            let pretty: Vec<String> =
                thread_counts.iter().map(|n| n.to_string()).collect();
            println!("  Sweep:      {}", pretty.join(", "));
        } else {
            println!("  Threads:    {}", thread_counts[0]);
        }
        println!(
            "  Iterations: {} per thread{}",
            args.iterations,
            if thread_counts.len() > 1 { " per run" } else { "" },
        );
        println!("  Warmup:     {} per thread", args.warmup);
    }

    let mut results: Vec<RunResult> = Vec::with_capacity(thread_counts.len());
    for (idx, &n_threads) in thread_counts.iter().enumerate() {
        if human {
            println!();
            if thread_counts.len() > 1 {
                println!(
                    "{}▶ Run {}/{}: {} thread{}{}",
                    style.bold,
                    idx + 1,
                    thread_counts.len(),
                    n_threads,
                    if n_threads == 1 { "" } else { "s" },
                    style.reset,
                );
            }
            if stdout_tty {
                print!("  Running…");
                io::stdout().flush().ok();
            }
        }
        let result =
            run_bench(n_threads, args.iterations, args.warmup, &cpath, &argv)?;
        if human {
            if stdout_tty {
                print!("\r\x1b[2K");
                io::stdout().flush().ok();
            }
            print_run(&result, style);
        }
        results.push(result);
    }

    if human {
        if results.len() > 1 {
            print_sweep_summary(&results, style);
            print_throughput_plot(&results, style);
        }
    } else {
        print_csv(&results);
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("spawnado: {e}");
            ExitCode::FAILURE
        }
    }
}
