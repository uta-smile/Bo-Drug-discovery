"""Integrate independently validated full crossed-control results into the open source."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]

def replace_block(text,start,end,replacement):
 a=text.index(start);b=text.index(end,a+len(start))
 return text[:a]+replacement+"\n\n"+text[b:]

def main():
 report=json.loads((ROOT/'results/crossed_auxiliary_v1/full/summary.json').read_text())
 assert report['stage']=='full' and report['audit']['clean'] and report['audit']['cases']==768
 assert report['audit']['max_legacy_recall_difference']==0
 e=report['estimates']
 paper=ROOT/'report/paper.tex';s=paper.read_text()
 if r'\input{generated/macros_crossed}' not in s:
  s=s.replace(r'\input{generated/macros_permutation}',r'\input{generated/macros_permutation}'+'\n'+r'\input{generated/macros_crossed}')
 s=s.replace('Within-target response permutation preserves the auxiliary activity\ndistributions while disrupting their correspondence with compounds.',
  'Within-target permutation preserves the response distributions actually\nobserved in auxiliary history. Crossed controls exchange conditioning\nresponses while holding response-informed model fits fixed.')
 # Describe the verified comparison without assuming permutation removes all gain.
 s=s.replace('six families, and this benefit largely disappears after permutation,\nsupporting a contribution from the measured auxiliary responses.',
  'six families. Its recall advantage is '+('reduced' if e['primary']['mean']>0 else 'not consistently reduced')+' after permutation,\n'+('supporting a contribution from' if e['primary']['mean']>0 else 'limiting evidence for')+' the measured auxiliary response correspondence.')
 s=s.replace('We combine response permutation, covariance-free joint fitting,\nuncertainty-adjusted acquisition, and historical-selection controls into a',
  'We combine historical-response permutation, crossed response/fitting\ninterventions, uncertainty-adjusted acquisition, and historical-selection\ncontrols into a')
 s=s.replace('within-target response permutation largely removes this benefit while\npreserving historical observation locations and auxiliary activity\ndistributions.',
  'permutation '+('reduces' if e['primary']['mean']>0 else 'does not consistently reduce')+' this benefit while preserving historical observation\nlocations and the observed auxiliary response distributions.')
 s=s.replace('Hvarfner et al.~\\cite{hvarfner2026pitfalls} showed that task relationships',
  'HyperBO learns GP priors from earlier tasks~\\cite{wang2024hyperbo},\nillustrating that auxiliary responses can inform fitting as well as\nposterior conditioning. Hvarfner et al.~\\cite{hvarfner2026pitfalls} showed that task relationships')
 s=s.replace('We examine a complementary question: whether gains in sequential compound\ndiscovery require the original auxiliary activity--compound correspondence.',
  'Our contribution is a diagnostic comparison of these routes in sequential\nmolecular screening, rather than a new permutation test or GP prior-learning\nalgorithm.')
 s=s.replace("each auxiliary target's empirical activity distribution but breaks its\nactivity--compound correspondence.",
  "the empirical distribution of the responses observed in each auxiliary\ntarget's history, but breaks their activity--compound correspondence.\nHistorical locations are drawn first; only their observed responses are\npermuted, with one fixed permutation per target and repeat.")
 c2=r'''\paragraph{C2. Response correspondence and joint fitting.}
At each selection round, we fit two multi-task donors at the same primary
observations and historical locations: one uses measured auxiliary responses
($M$), the other their historical-response permutation ($P$). We cross the
parameter donor with the responses used for posterior conditioning, yielding
$MM$, $MP$, $PM$, and $PP$; the first letter denotes the parameter donor.
Coupling, signal variance, noise ratio, all normalization transforms, and
numerical jitter are fixed during reconditioning. Donors are re-estimated
at every round from the observations available on that policy's trajectory.
Thus, $MM$ and $PP$ reproduce ordinary measured- and permuted-response
multi-task BO. At a shared state, $MM$ versus $MP$ changes response
correspondence without changing predictive variance, whereas $MM$ versus
$PM$ changes fitting while retaining measured conditioning responses.

We evaluate the four policies both as sequential campaigns and as next-batch
replays on shared single-task and $MM$ trajectories. The latter hold the
primary observations fixed; the former include feedback from differing
selections. Parameters learned from measured responses still carry response
information, so $MP$ is not an auxiliary-value-free control. The earlier
covariance-free check (\mtzero{}) remains a special case: fixing $r=0$ and
all appropriate single-task parameters reproduces the single-task posterior.
Supplement~S2 gives implementation details and the original JAK comparison.'''
 s=replace_block(s,r'\paragraph{C2. Covariance-free joint fitting.}',r'\paragraph{C3.',c2)
 # Replace the existing C1 evidence with the independently checked new null.
 a=s.index('Under uniform histories, the benefit',s.index(r'\section{Results}'))
 b=s.index('The withheld-history experiments show',a)
 c1=r'''Under uniform histories, fitted multi-task BO ($MM$) improves recall
relative to \stl{} by $\crossMeasuredGain{}$ percentage points, compared with
$\crossPermutedGain{}$ after permutation of the observed historical responses
($PP$). The paired measured-minus-permuted contrast is $\crossPrimary{}$
points (95\% CI $\crossPrimaryCi{}$), positive in
$\crossPrimaryPositive{}/6$ families ($p=\crossPrimaryP{}$).
Historical locations and the exact response multiset for each auxiliary
target are unchanged. This comparison therefore tests response correspondence
without changing the observed response distributions. Each condition uses
\crossCampaigns{} campaigns per family and budget; inference averages the
two budgets within each family. The new runs reproduce the original
single-task and measured-response recall vectors exactly. Per-budget values
and legacy permutation diagnostics are reported in Supplement~S2.

'''
 s=s[:a]+c1+s[b:]
 new_c2=r'''\subsection{Crossed Controls Separate Conditioning from Response-Informed Fitting}
\label{sec:res-mech}

Table~\ref{tab:crossed} compares the crossed policies under uniform histories.
With the measured-response fit retained, replacing measured conditioning
responses by their permutation changes recall by
$\crossResponseMeasuredFit{}$ points ($MM-MP$; 95\% CI
$\crossResponseMeasuredFitCi{}$). With the permuted-response fit retained,
the corresponding contrast is $\crossResponsePermutedFit{}$ points
($PM-PP$; $\crossResponsePermutedFitCi{}$). Conversely, changing the fit
while retaining measured responses gives $\crossFitMeasuredResponse{}$
points ($MM-PM$; $\crossFitMeasuredResponseCi{}$), and with permuted
responses gives $\crossFitPermutedResponse{}$ points
($MP-PP$; $\crossFitPermutedResponseCi{}$).

\begin{table}[htbp]
\caption{C2: Recall@10 gain over \stl{} (percentage points) under uniform
histories, averaged over the two budgets within each family. The first
letter identifies the response condition supplying fitted parameters; the
second identifies the conditioning responses ($M$: measured, $P$: permuted
within observed history). Each budget uses \crossCampaigns{} campaigns.}
\label{tab:crossed}
\centering
\input{generated/table_crossed}
\end{table}

These are predefined descriptive policy contrasts, not independent tests
of biological mechanisms. Their factorial interaction is
$\crossInteraction{}$ points (95\% CI $\crossInteractionCi{}$); the
conditional comparisons need not agree, and the effects are not assigned
causal percentages. Matched-history replay provides a complementary check:
on shared single-task trajectories, $MM$ and $MP$ recover
$\crossStlAnchorMMBatchHits{}$ and $\crossStlAnchorMPBatchHits{}$ top-10
compounds per proposed batch, respectively; on shared $MM$ trajectories,
the corresponding means are $\crossMtAnchorMMBatchHits{}$ and
$\crossMtAnchorMPBatchHits{}$. Each pair has identical predictive variance
at every shared state. Any difference in these proposed batches therefore
comes from the response-dependent predictive means at the fixed fit,
not from a variance change. Replay outcomes describe one-step choices,
not a decomposition of final sequential recall. The original covariance-free
JAK result remains supporting evidence in Supplement~S2.'''
 s=replace_block(s,r'\subsection{Shared Hyperparameter Fitting Reproduces the JAK Gain}',
                 r'\subsection{Acquisition Uncertainty',new_c2)
 s=s.replace('consistent with useful activity information, although the comparison\ndoes not isolate its contribution exactly.',
  'consistent with useful activity information, although the comparison\ndoes not isolate its contribution exactly. Uncertainty scaling is an\nacquisition intervention, not evidence of improved calibration; existing\ncalibration and chemical-isolation probes are reported in Supplement~S2.')
 s=s.replace('% Supporting table moved to supplement: tab:exploration-control','% Detailed uncertainty and calibration diagnostics are in Supplement S2.')
 # Keep the existing C4 results, replacing only the duplicated budget-cell grid.
 ta=s.index(r'\begin{table}',s.index(r'\paragraph{Effects across protein families.}'))
 tb=s.index(r'\end{table}',ta)+len(r'\end{table}')
 c4=r'''\begin{table}[htbp]
\caption{C4: Recall@10 gain over \stl{} (percentage points), averaged over
the two budgets within each family under withheld and uniform histories.
Coverage is the fraction of library compounds included in auxiliary history.
The twelve budget rows are retained in Supplement~S2.1.}
\label{tab:design}
\centering\small
\input{generated/table_design_family}
\end{table}'''
 s=s[:ta]+c4+s[tb:]
 s=s.replace('More broadly, an evaluation must distinguish\nuseful activity information from changes in fitting or compound selection.',
  'Auxiliary response information can enter through posterior conditioning\nor response-informed fitting. Both routes should be distinguished from\ngains reproduced by uninformative responses or outcome-informed historical\nselection.')
 s=s.replace('at matched historical observation locations to assess the contribution of\nthe original activity information.',
  'at matched historical observation locations, preserving the observed\nresponse multisets, to assess the contribution of activity correspondence.\nThen cross response conditions with locked response-informed fits to compare\nconditioning and fitting at shared states and in sequential campaigns.')
 s=s.replace('The controls reproduce gains of the observed magnitude without\nquantifying each mechanism\'s contribution to an individual gain.',
  'The controls reproduce gains of the observed magnitude without\nquantifying each mechanism\'s contribution to an individual gain. Sequential\ncrossed-policy contrasts also include differences in acquired primary\nobservations; matched-history replays isolate one-step decisions instead.')
 s=s.replace('discovery performance with response-permutation and mechanism controls to\ninterpret the contribution of historical assay data.',
  'discovery performance with response-permutation and crossed fitting controls\nto distinguish response correspondence at a fixed fit from response-informed\nparameter estimation, while checking historical-sampling sensitivity.')
 paper.write_text(s,encoding='utf8')

 supp=ROOT/'report/supplement.tex';u=supp.read_text()
 if r'\input{generated/macros_crossed}' not in u:
  u=u.replace(r'\input{generated/macros_permutation}',r'\input{generated/macros_permutation}'+'\n'+r'\input{generated/macros_crossed}')
 u=u.replace('Table~2, without the family-level summary rows.','Table~3, without the family-level summary rows.')
 u=u.replace(r'\subsection{The six-family null comparison, per family}',r'\subsection{Legacy six-family null comparisons}')
 u=u.replace('Table~\\ref{tab:permutation} gives the family-level values behind the headline\nresult of the main text.',
  'Table~\\ref{tab:permutation} retains the earlier six-family permutation\nresults for sensitivity analysis. Legacy records do not document\nwhether permutation preceded history selection; these legacy results must\nnot be interpreted as preserving each observed history\'s response multiset.\nThe main text instead uses the verified historical-response permutation\nand crossed-control runs described below.')
 u=u.replace('uniformly sampled histories, with measured auxiliary responses and with\nresponses permuted within each target. The permutation holds the observation\nlocations fixed and preserves each auxiliary target\'s marginal activity\ndistribution, so the \\emph{difference} column isolates the contribution of the\nactivity--compound correspondence.',
  'uniform histories in the legacy measured and permuted run classes.\nHistorical observation locations are matched. Preservation of the response\ndistribution within the observed historical sample was not independently\nverified for this run class; it is not the new main-text null.')
 # Legacy null comparison no longer licenses distribution-shape attribution.
 a=u.index('Comparing the two nulls separates the correspondence')
 b=u.index(r'\paragraph{Covariance-free joint fitting (C2).}',a)
 u=u[:a]+r'''The legacy measured-minus-permuted and measured-minus-Gaussian contrasts
are retained as recorded, rather than relabeled as the newly verified null.
Their difference is \permMtNullDifference{} points for \mt{}
(95\% CI \permCiMtNullDifference{}) and \permPooledNullDifference{}
for \pooled{} (\permCiPooledNullDifference{}). This comparison alone does
not establish that observed-history distributional effects are absent.

'''+u[b:]
 crossed=r'''\subsection{Verified historical-response permutation and crossed fits}
\label{app:crossed}

The new null draws historical locations first, then permutes the fifty
observed responses within each auxiliary target, separately for each held-out
primary target and repeat. Permutations remain fixed through each campaign.
The four crossed conditions are defined in main-text C2; every locked fit
retains the donor's coupling, noise-to-signal ratio, signal variance, per-task
means and scales, inner GP normalization, and numerical jitter. Reconditioning
reuses the donor's Cholesky factor and changes only the response coefficients.
For fixed covariance $K_\theta$, noise $\Lambda_\theta$, and test covariance
$k_*$, the posterior variance
\[
v_*=k_{**}-k_*^\top(K_\theta+\Lambda_\theta)^{-1}k_*
\]
does not depend on conditioning responses. This is a standard GP property;
the contribution is its controlled use to distinguish routes of auxiliary
response information in molecular screening, not a new variance identity.

All six panels use the original two budgets, four primary targets, and
32 repeats per target. A larger-budget trajectory supplies the smaller-budget
prefix, reproducing the deterministic truncated-batch policy exactly.
The two-repeat smoke phase checks all panels and budgets without selecting
experiments by their effect sizes. Same-condition locked posteriors reproduce
their donors, and locked predictive variances are exactly unchanged under
response replacement. Independent reconstruction confirms zero discrepancy
from the stored single-task and measured-response recall vectors at both
budgets. The observed response multisets are identical; moment discrepancies
are at floating-point rounding level. Selected indices, permutation source
indices, input hashes, parameter snapshots, and replay records are saved in
\texttt{results/crossed\_auxiliary\_v1/}.

The primary sequential comparison is $MM-PP$: $\crossPrimary{}$ points,
95\% CI $\crossPrimaryCi{}$, positive in $\crossPrimaryPositive{}/6$
families, with two-sided Wilcoxon $p=\crossPrimaryP{}$.
Other conditional contrasts and their interaction are descriptive. Budgets,
primary targets, and repeated seeds do not create additional independent
family observations. Parameters supplied by the measured donor retain
response information even when conditioning responses are permuted.

Replays evaluate all policies at every batch boundary on shared \stl{}
and $MM$ trajectories, with common candidates, incumbents, and tie-breaking.
Future primary activities score the proposed batches retrospectively and
never enter the fitting function. Rounds are averaged within campaigns,
then budgets and campaigns within each family. These summaries describe
one-step selection on fixed histories, while sequential-policy contrasts
include feedback from differing acquired observations. They do not identify
unique causal mechanisms or assign causal fractions to final recall.

\begin{table}[htbp]
\centering\small
\input{generated/table_crossed}
\caption{Verified crossed-control recall gains over single-task BO, in
percentage points. Each family averages its two budgets; condition notation
matches main-text Table~2. The complete per-budget outcomes and additional
replay metrics are in the independent summary JSON.}
\label{tab:crossed-supp}
\end{table}

'''
 u=u.replace(r'\subsection{Uncertainty trajectory behind the C3 scale}',crossed+r'\subsection{Uncertainty trajectory behind the C3 scale}')
 u=u.replace('reported in the main text as an upper bound. The JAK scale',
  'reported in the main text as an upper bound. This comparison does not\nestablish improved uncertainty calibration. Existing held-out residual\nprobes give standardized-residual standard deviations above one on both\nJAK and CYP, rather than the approximately 0.80 value expected from a\ncalibration explanation of JAK\'s compression. Changed JAK selections also\nshow no clear separation in chemical isolation. These are supporting\ndiagnostics, not causal identifications (\\texttt{results/mechanism\\_probes.json}).\nThe JAK scale')
 supp.write_text(u,encoding='utf8')
 print('Integrated full audited results into paper.tex and supplement.tex')

if __name__=='__main__':main()
