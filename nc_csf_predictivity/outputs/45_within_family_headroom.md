# Within-family headroom and commitment-criteria checks (descriptive, 2026-09-15)

## (1) Exhaustive raw-confidence threshold audit, ResNet-18, fallback CTM

- none_nr_marginal: best policy with nonzero coverage 2.8197 at checkpoint coverage 0.018; all-fallback CTM 2.7901; distinct confidences 56
- source_nr_marginal: best policy with nonzero coverage 2.8461 at checkpoint coverage 0.018; all-fallback CTM 2.7901; distinct confidences 56

## (2) Probe pool with a hindsight fixed detector per encoder

- probe_clip_vitb16: hindsight fixed ViM at 168.3030 over 160 rows
- probe_dinov2_vitb14: hindsight fixed NCI at 165.4951 over 160 rows
- decomposition 10.8260 = 5.1666 (CTM to pool fixed ViM) + 2.0824 (pool fixed to encoder fixed) + 0.6833 (encoder fixed to checkpoint oracle) + 2.8937 (checkpoint to row oracle)

## (3) VGG-13 source-held-out: new ID dataset within a represented family

| Held-out source | Rows | Training rule (other sources) | Regret | Metadata rule regret | Hindsight fixed | Regret | Checkpoint-oracle regret | Ceiling over training rule |
|---|---:|---|---:|---:|---|---:|---:|---:|
| cifar10 | 480 | CTM | 9.31 | 9.31 | Confidence | 1.79 | 0.56 | 8.75 |
| cifar100 | 560 | CTM | 3.45 | 3.91 | CTM | 3.45 | 1.68 | 1.77 |
| supercifar100 | 720 | NNGuide | 10.98 | 10.98 | CTM | 8.03 | 4.00 | 6.97 |
| tinyimagenet | 480 | NNGuide | 3.76 | 7.08 | CTM | 0.50 | 0.44 | 3.32 |

Caveats: four sources of which CIFAR-100 and SuperCIFAR-100 share images; the 'Confidence' detector is each paradigm's learned readout; development data inspected many times.
