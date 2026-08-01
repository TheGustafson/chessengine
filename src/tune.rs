//! The `tunables!` macro behind the SPSA loop (D45, design/14 §9.6).
//!
//! `src/search/params.rs` declares every SPSA-movable scalar through one
//! invocation of this macro. In the DEFAULT build each entry expands to an
//! `#[inline(always)]` getter returning a literal — LLVM folds it exactly
//! like the `pub const` it replaced, so the shipped engine and the depth-6
//! bench are bit-identical (D24 guards this). Under `--features tune` each
//! entry is backed by a process-global relaxed atomic, settable at runtime
//! via `setoption name <NAME> value <x>` (the hook in `uci.rs`), which is
//! what lets one binary play both sides of an SPSA perturbation pair.
//!
//! Getters keep the SCREAMING_SNAKE names of the constants they replaced
//! (`#[allow(non_snake_case)]`): the whole point of the D20 inventory is
//! greppability, and `params::LMR_DEEPER_MARGIN()` still reads as the
//! named constant at every use site.
//!
//! Values read once at construction time (e.g. the reduction table built in
//! `Worker::new` from `RED_TABLE_NUM`) pick up overrides only if `setoption`
//! arrives first — true under fastchess, which configures options before the
//! first `go`. Options are constant for a process's lifetime in the SPSA
//! harness, so no invalidation machinery exists on purpose.

/// See the module docs. Entry syntax: `pub NAME: Type = default;` where
/// `Type` is a scalar (`Value`, `Depth`, `i32`, `i64`, `u64`, `usize`) or
/// `f64`. Non-scalars (tuples, arrays) stay ordinary `pub const`s outside
/// the invocation.
#[macro_export]
macro_rules! tunables {
    ( $( $(#[$meta:meta])* pub $name:ident : $ty:tt = $default:expr; )* ) => {
        $( $crate::tunables!(@getter $(#[$meta])* $name : $ty = $default); )*

        #[cfg(feature = "tune")]
        #[allow(non_upper_case_globals)]
        mod atoms {
            $( $crate::tunables!(@atom $name : $ty = $default); )*
        }

        /// Set a tunable by name from a raw option string. Unknown names and
        /// unparsable values are loud errors — a silent no-op here would let
        /// an SPSA driver typo tune nothing while reporting success.
        #[cfg(feature = "tune")]
        pub fn set_tunable(name: &str, raw: &str) -> Result<(), String> {
            match name {
                $( stringify!($name) => $crate::tunables!(@store $name : $ty, raw), )*
                _ => Err(format!("unknown tunable '{name}'")),
            }
        }

        /// Every tunable as `(name, current-value-as-string)`, in
        /// declaration order. The SPSA driver preflights its config here.
        #[cfg(feature = "tune")]
        pub fn tunables_list() -> Vec<(&'static str, String)> {
            vec![ $( (stringify!($name), format!("{}", $name())) ),* ]
        }
    };

    (@getter $(#[$meta:meta])* $name:ident : f64 = $default:expr) => {
        #[cfg(not(feature = "tune"))]
        $(#[$meta])*
        #[allow(non_snake_case)]
        #[inline(always)]
        pub fn $name() -> f64 { $default }

        #[cfg(feature = "tune")]
        $(#[$meta])*
        #[allow(non_snake_case)]
        #[inline(always)]
        pub fn $name() -> f64 {
            f64::from_bits(atoms::$name.load(::core::sync::atomic::Ordering::Relaxed))
        }
    };
    (@getter $(#[$meta:meta])* $name:ident : $ty:tt = $default:expr) => {
        #[cfg(not(feature = "tune"))]
        $(#[$meta])*
        #[allow(non_snake_case)]
        #[inline(always)]
        pub fn $name() -> $ty { $default }

        #[cfg(feature = "tune")]
        $(#[$meta])*
        #[allow(non_snake_case)]
        #[inline(always)]
        pub fn $name() -> $ty {
            atoms::$name.load(::core::sync::atomic::Ordering::Relaxed) as $ty
        }
    };

    (@atom $name:ident : f64 = $default:expr) => {
        pub static $name: ::core::sync::atomic::AtomicU64 =
            ::core::sync::atomic::AtomicU64::new(f64::to_bits($default));
    };
    (@atom $name:ident : $ty:tt = $default:expr) => {
        pub static $name: ::core::sync::atomic::AtomicI64 =
            ::core::sync::atomic::AtomicI64::new(($default) as i64);
    };

    (@store $name:ident : f64, $raw:ident) => {{
        let v: f64 = $raw
            .trim()
            .parse()
            .map_err(|e| format!("bad f64 for {}: {e}", stringify!($name)))?;
        atoms::$name.store(v.to_bits(), ::core::sync::atomic::Ordering::Relaxed);
        Ok(())
    }};
    (@store $name:ident : $ty:tt, $raw:ident) => {{
        let v: i64 = $raw
            .trim()
            .parse()
            .map_err(|e| format!("bad int for {}: {e}", stringify!($name)))?;
        atoms::$name.store(v, ::core::sync::atomic::Ordering::Relaxed);
        Ok(())
    }};
}

/// Constified handles (D50, 2026-08-01): the same `NAME()` getter spelling as
/// [`tunables!`], but the value is a compile-time constant in EVERY build — a
/// `tune` build does not expose it, so it is off the SPSA surface entirely.
///
/// D50 reverses design/01 §3.8's default: **a constant becomes a tunable when
/// a campaign wants it, not by default.** Promoting one back is a one-line
/// move into the `tunables!` block (call sites are untouched either way).
#[macro_export]
macro_rules! constants {
    ( $( $(#[$meta:meta])* pub $name:ident : $ty:tt = $default:expr; )* ) => {
        $(
            $(#[$meta])*
            #[allow(non_snake_case)]
            #[inline(always)]
            pub fn $name() -> $ty { $default }
        )*
    };
}
