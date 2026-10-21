//! Consensus densification statistics (walls-first plan section 8).
//!
//! The error-map growth signal used to be the window-MAX over views of a
//! gaussian's coverage-weighted mean error, so a gaussian that one photo
//! (a lighting-inconsistent one, say) disliked became a growth candidate. This
//! replaces MAX by a vote: in each view, among the gaussians the view COVERS,
//! the worst `q` fraction by rank are "hits"; a gaussian is a candidate only
//! when enough views, and enough of the views that saw it, agree.
//!
//! Host-side and pure so it is testable without a GPU. A rank is invariant to
//! each view's error scale, so a photo that is wrong everywhere (exposure) still
//! marks only its worst `q` and cannot flood the counters.

/// Per-gaussian vote counters for one refine window.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ConsensusStats {
    /// Views in which the gaussian was a hit.
    cnt: Vec<f32>,
    /// Views that covered the gaussian.
    vis: Vec<f32>,
    /// Sum of the gaussian's error over the views in which it was a hit.
    hit_e: Vec<f32>,
}

/// A gaussian with `sum_p T alpha` at or below this is not covered by the view.
/// A GUESS: the plan says "> epsilon" without a value. 1e-3 of one fully opaque
/// pixel is far below anything the rasteriser draws.
pub(crate) const COVER_EPS: f32 = 1e-3;

/// How many of `n_covered` gaussians are hits at quantile `q`: `ceil(q n)`,
/// at least one. Equivalent to the plan's `rank/N > 1 - q` with 1-based
/// ascending rank, written in integers so float noise cannot move the cut.
pub(crate) fn hit_count(n_covered: usize, q: f32) -> usize {
    if n_covered == 0 {
        return 0;
    }
    // `q` is an f32: 0.1f32 is 0.10000000149..., and times 10 that would round
    // UP to 2 hits. Snap it to 6 decimals first, then absorb the f64 product's
    // own noise with a tiny epsilon.
    let q = (f64::from(q) * 1e6).round() / 1e6;
    let exact = q * n_covered as f64;
    ((exact - 1e-7).ceil() as usize).clamp(1, n_covered)
}

impl ConsensusStats {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            cnt: vec![0.0; n],
            vis: vec![0.0; n],
            hit_e: vec![0.0; n],
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.cnt.len()
    }

    /// Fold one view in. `e` is the gaussian's coverage-weighted mean error in
    /// this view and `cover` its `sum_p T alpha`. The hit set is the top
    /// `hit_count(n_covered, q)` COVERED gaussians by `e` (ties broken by lower
    /// index, so the result is deterministic), and only those with `e > 0`: a
    /// rank among all-zero errors is not a vote.
    pub(crate) fn add_view(&mut self, e: &[f32], cover: &[f32], q: f32) {
        assert_eq!(e.len(), self.len(), "one error per gaussian");
        assert_eq!(cover.len(), self.len(), "one coverage per gaussian");
        let covered: Vec<usize> = (0..self.len()).filter(|&g| cover[g] > COVER_EPS).collect();
        for &g in &covered {
            self.vis[g] += 1.0;
        }
        let mut order = covered;
        // Descending by error, ascending by index on ties. NaN sorts last.
        order.sort_by(|&a, &b| {
            let (ea, eb) = (finite_or_neg(e[a]), finite_or_neg(e[b]));
            eb.total_cmp(&ea).then(a.cmp(&b))
        });
        let m = hit_count(order.len(), q);
        for &g in order.iter().take(m) {
            if e[g].is_finite() && e[g] > 0.0 {
                self.cnt[g] += 1.0;
                self.hit_e[g] += e[g];
            }
        }
    }

    /// Candidate iff `cnt >= k` AND `cnt >= frac * vis`.
    pub(crate) fn candidates(&self, k: u32, frac: f32) -> Vec<bool> {
        (0..self.len())
            .map(|g| self.cnt[g] >= k as f32 && self.cnt[g] >= frac * self.vis[g])
            .collect()
    }

    /// Sampling weight `(cnt / vis) * mean error over the hit views`; 0 where
    /// the gaussian was never covered or never hit.
    pub(crate) fn weights(&self) -> Vec<f32> {
        (0..self.len())
            .map(|g| {
                if self.vis[g] <= 0.0 || self.cnt[g] <= 0.0 {
                    0.0
                } else {
                    (self.cnt[g] / self.vis[g]) * (self.hit_e[g] / self.cnt[g])
                }
            })
            .collect()
    }

    /// Re-index after a prune: keep `idx` (ascending, as `argwhere` returns).
    pub(crate) fn select(&self, idx: &[usize]) -> Self {
        let pick = |v: &Vec<f32>| idx.iter().map(|&i| v[i]).collect();
        Self {
            cnt: pick(&self.cnt),
            vis: pick(&self.vis),
            hit_e: pick(&self.hit_e),
        }
    }

    #[cfg(test)]
    pub(crate) fn counts(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.cnt, &self.vis, &self.hit_e)
    }
}

fn finite_or_neg(x: f32) -> f32 {
    if x.is_finite() { x } else { f32::NEG_INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: f32 = 0.10;
    const K: u32 = 3;
    const PHI: f32 = 0.5;

    // Gaussians: A=0 B=1 C=2 D=3 E=4 F=5, 8 views.
    //
    //   A  covered only in v6, v7 and the top of both       -> cnt 2, vis 2 (k fails)
    //   B  covered in v0..v4, the top of v0, v1, v2         -> cnt 3, vis 5 (candidate)
    //   C  covered in all 8, the top of v3, v4, v5          -> cnt 3, vis 8 (phi fails: 3 < 4)
    //   D  never covered, but carries a huge error value    -> never anything
    //   E,F covered in all 8, never the top
    //
    // n_covered <= 5 in every view, so the hit set is the single worst gaussian.
    // Under window-MAX every gaussian with a peak is a candidate: {A, B, C}.
    fn views() -> Vec<(Vec<f32>, Vec<f32>)> {
        const BIG: f32 = 9.0;
        let (a, b, c, d, e, f) = (0, 1, 2, 3, 4, 5);
        let mut out = Vec::new();
        for v in 0..8 {
            let mut err = vec![0.0f32; 6];
            let mut cov = vec![1.0f32; 6];
            cov[d] = 0.0;
            err[d] = 50.0; // an uncovered row's value must never matter
            if v > 4 {
                cov[b] = 0.0;
            }
            if v < 6 {
                cov[a] = 0.0;
            }
            // Background error for non-top covered gaussians: small, distinct.
            for (g, base) in [(a, 0.5), (b, 0.4), (c, 0.3), (e, 0.6), (f, 0.7)] {
                err[g] = base;
            }
            match v {
                0..=2 => err[b] = BIG,
                3..=5 => err[c] = BIG,
                6 | 7 => err[a] = BIG,
                _ => unreachable!(),
            }
            // Scale one view by 1000: ranks, hence votes, must not change.
            if v == 5 {
                for x in &mut err {
                    *x *= 1000.0;
                }
            }
            out.push((err, cov));
        }
        out
    }

    fn run(views: &[(Vec<f32>, Vec<f32>)], n: usize) -> ConsensusStats {
        let mut s = ConsensusStats::new(n);
        for (e, c) in views {
            s.add_view(e, c, Q);
        }
        s
    }

    fn cands(s: &ConsensusStats, k: u32, phi: f32) -> Vec<usize> {
        s.candidates(k, phi)
            .iter()
            .enumerate()
            .filter_map(|(g, c)| c.then_some(g))
            .collect()
    }

    /// Plan test 19. The candidate set is exactly {B}.
    #[test]
    fn consensus_selects_only_the_gaussian_a_majority_of_its_views_dislike() {
        let s = run(&views(), 6);
        let (cnt, vis, _) = s.counts();
        assert_eq!(cnt, [2.0, 3.0, 3.0, 0.0, 0.0, 0.0]);
        assert_eq!(vis, [2.0, 5.0, 8.0, 0.0, 8.0, 8.0]);
        assert_eq!(cands(&s, K, PHI), [1], "B only");

        // Separation: each ingredient of the rule rejects something here.
        let max_rule: Vec<usize> = (0..6).filter(|&g| s.counts().0[g] >= 1.0).collect();
        assert_eq!(
            max_rule,
            [0, 1, 2],
            "MAX-like aggregation admits {{A, B, C}}"
        );
        assert_eq!(cands(&s, K, 0.0), [1, 2], "without phi, C joins");
        assert_eq!(
            cands(&s, 1, PHI),
            [0, 1],
            "without k, A (2 of its 2 views) joins"
        );
        assert_eq!(
            cands(&s, 1, 0.0),
            [0, 1, 2],
            "k=1, phi=0 is the per-view-rank any-hit rule"
        );
    }

    /// A was a hit in BOTH views that covered it: it passes phi (2 >= 0.5*2) and
    /// is rejected by k alone. Dropping k must change the answer.
    #[test]
    fn k_is_load_bearing_independently_of_phi() {
        let s = run(&views(), 6);
        let a_passes_phi = s.candidates(1, PHI)[0];
        assert!(a_passes_phi, "A passes phi=0.5 with k=1");
        assert!(!s.candidates(K, PHI)[0], "and is rejected only by k=3");
    }

    /// Weight `(cnt/vis) * mean hit error`.
    #[test]
    fn weights_are_vote_fraction_times_mean_hit_error() {
        let s = run(&views(), 6);
        let w = s.weights();
        assert!((w[1] - (3.0 / 5.0) * 9.0).abs() < 1e-6, "B: {}", w[1]);
        // C's three hits: v3, v4 at 9, v5 at 9000 (that view is scaled x1000).
        assert!(
            (w[2] - (3.0 / 8.0) * ((9.0 + 9.0 + 9000.0) / 3.0)).abs() < 1e-3,
            "C: {}",
            w[2]
        );
        assert_eq!(w[3], 0.0, "never covered");
        assert_eq!(w[4], 0.0, "covered, never hit");
    }

    /// The same data twice gives identical counters and weights (no hash order,
    /// no unstable sort).
    #[test]
    fn consensus_is_deterministic() {
        let v = views();
        let (a, b) = (run(&v, 6), run(&v, 6));
        assert_eq!(a, b);
        assert_eq!(a.weights(), b.weights());
        assert_eq!(a.candidates(K, PHI), b.candidates(K, PHI));
    }

    /// A rank is invariant to a view's error scale: x1000 on every view leaves
    /// the votes untouched (only the weights scale).
    #[test]
    fn votes_do_not_depend_on_the_error_scale() {
        let v = views();
        let scaled: Vec<_> = v
            .iter()
            .map(|(e, c)| (e.iter().map(|x| x * 1000.0).collect(), c.clone()))
            .collect();
        assert_eq!(run(&v, 6).counts().0, run(&scaled, 6).counts().0);
        assert_eq!(run(&v, 6).counts().1, run(&scaled, 6).counts().1);
    }

    /// 12 gaussians; in 3 views gaussians 0..10 are covered (10 covered, 2 not).
    /// Gaussian 1 is the worst, gaussian 0 second worst, every time. The hit
    /// count is ceil(0.1 * 10) = 1 over the COVERED rows. Taking the quantile
    /// over ALL 12 rows gives ceil(0.1 * 12) = 2 and promotes gaussian 0.
    #[test]
    fn the_quantile_is_taken_over_covered_rows_only() {
        let mut s = ConsensusStats::new(12);
        for _ in 0..3 {
            let mut e = vec![0.0f32; 12];
            let mut cov = vec![0.0f32; 12];
            for g in 0..10 {
                cov[g] = 1.0;
                e[g] = 1.0;
            }
            e[1] = 10.0;
            e[0] = 9.0;
            s.add_view(&e, &cov, Q);
        }
        assert_eq!(s.counts().0[1], 3.0);
        assert_eq!(s.counts().0[0], 0.0, "second-worst is not a hit");
        assert_eq!(cands(&s, K, PHI), [1]);
        assert_eq!(hit_count(10, Q), 1);
        assert_eq!(
            hit_count(12, Q),
            2,
            "the all-rows count differs on this fixture"
        );
    }

    #[test]
    fn hit_count_edges() {
        assert_eq!(hit_count(0, Q), 0);
        assert_eq!(hit_count(1, Q), 1, "at least one");
        assert_eq!(
            hit_count(10, 0.1),
            1,
            "q*n an exact integer does not round up"
        );
        assert_eq!(hit_count(11, 0.1), 2);
        assert_eq!(hit_count(5, 1.0), 5);
        // The f32 value of the DEFAULT quantile must give the integer answer.
        assert_eq!(hit_count(30, 0.1f32), 3);
        assert_eq!(hit_count(1_000_000, 0.1f32), 100_000);
    }

    /// A view where every covered error is 0 casts no vote (a tie among zeros
    /// is not evidence), and non-finite errors never vote.
    #[test]
    fn zero_and_nonfinite_errors_cast_no_vote() {
        let mut s = ConsensusStats::new(3);
        s.add_view(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], Q);
        s.add_view(&[f32::NAN, f32::INFINITY, 0.0], &[1.0, 1.0, 1.0], Q);
        assert_eq!(s.counts().0, [0.0, 0.0, 0.0]);
        assert_eq!(s.counts().1, [2.0, 2.0, 2.0], "they were still covered");
    }

    #[test]
    fn select_reindexes_all_three_counters() {
        let s = run(&views(), 6);
        let kept = s.select(&[1, 2, 5]);
        assert_eq!(kept.len(), 3);
        assert_eq!(kept.counts().0, [3.0, 3.0, 0.0]);
        assert_eq!(kept.counts().1, [5.0, 8.0, 8.0]);
        assert_eq!(cands(&kept, K, PHI), [0]);
    }
}
