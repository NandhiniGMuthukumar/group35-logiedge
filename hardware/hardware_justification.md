# Hardware justification

Module 2 balances Compute Performance, Power Budget and Unit Cost [2]. The 10 W budget makes power the dominant constraint. Using the assignment's indicative figures, Pi 5 with AI HAT+ costs Rs 12.75 lakh for 85 trucks and Rs 39.75 lakh for 265 at 7.5 W. Jetson's 15 W exceeds budget. STM32 is inexpensive but requires a different embedded runtime and orchestration architecture.

For the assigned Roofline example, 45 MFLOP divided by 18 MB gives 2.5 FLOP/byte. The ridge is 16 GFLOP/s divided by 12 GB/s, or 1.333 FLOP/byte. Compute dominates, with an ideal 2.8125 ms bound. This hypothetical workload differs from our tiny measured MLP.

| Option | Rs/truck | Assignment power | Cost for 85 | Cost for 265 |
|---|---:|---:|---:|---:|
| Pi 5 8 GB and AI HAT+ | 15,000 | 7.5 W | Rs 12,75,000 | Rs 39,75,000 |
| Jetson Orin Nano Super | 45,000 | 15 W | Rs 38,25,000 | Rs 1,19,25,000 |
| STM32H7 custom design | 3,500 | 0.4 W | Rs 2,97,500 | Rs 9,27,500 |

The actual small MLP may fit an appropriate MCU's model storage, but this Python/TensorFlow/Docker/SQLite stack requires a different embedded implementation there. The chosen Pi has 8 GB RAM and microSD storage; model-file size alone cannot establish peak working memory or system image requirements. AI HAT peak TOPS is not measured application performance. These course prices/power numbers are planning inputs, not current quotes or measured total-device power.

References: see [project references](../docs/references.md).
