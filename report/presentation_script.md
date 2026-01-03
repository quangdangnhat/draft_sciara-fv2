# Presentation Script: CUDA Parallelization of Sciara-fv2
## Nhat Quang Dang - University of Calabria
## January 15, 2025

---

## Slide 1: Title Slide (30 giây)

**[Mở đầu - Chào hỏi]**

> "Good morning/afternoon everyone. My name is Nhat Quang Dang, registration number 279990, from the University of Calabria.
>
> Today I will present my work on **CUDA Parallelization of the Sciara-fv2 Lava Flow Simulator**, focusing on performance analysis and optimization strategies."

---

## Slide 2: Outline (20 giây)

**[Giới thiệu cấu trúc]**

> "Here is the outline of my presentation. I will cover:
> 1. The **context** and motivation
> 2. The **roadmap** of 5 CUDA optimization strategies
> 3. **Execution time** comparison
> 4. **Roofline model** analysis
> 5. **GPU occupancy** insights
> 6. **FLOP count** comparison
> 7. And finally, **conclusions** and key takeaways."

---

## Slide 3: Context (1 phút 30 giây)

**[Giải thích bài toán]**

> "Let me start with the context.
>
> **Sciara-fv2** is a Cellular Automata model used to simulate **lava flow** from volcanic eruptions. In this project, we simulate the **Mount Etna 2006 eruption** in Italy.
>
> The simulation grid has **517 by 378 cells**, giving us approximately **195,000 cells** in total. Each cell stores three main properties:
> - **Altitude** - the terrain height
> - **Lava thickness** - how much lava is present
> - **Temperature** - which affects viscosity
>
> **[Chỉ vào hình Moore Neighborhood]**
>
> The algorithm uses a **Moore neighborhood** - meaning each cell interacts with its **8 surrounding neighbors** plus itself, forming a 9-cell stencil pattern.
>
> **The challenge** is that this stencil computation is **memory-intensive**, requiring many memory accesses per cell. Our goal is to parallelize this on GPU - specifically the **NVIDIA GTX 980** with 2048 CUDA cores and 2MB L2 cache."

---

## Slide 4: Roadmap (1 phút 30 giây)

**[Giải thích các versions]**

> "Now let me explain the **5 CUDA optimization strategies** we implemented.
>
> We started with **Version 1: Global Memory** - this is our **baseline**. It directly accesses global memory without any optimization.
>
> From here, we explored **two branches**:
>
> **[Chỉ vào nhánh trái - Tiled]**
>
> The **left branch** focuses on **reducing memory latency** using shared memory:
> - **Version 2: Tiled** - loads data into shared memory in 16×16 tiles
> - **Version 3: Tiled+Halo** - adds a halo region to handle border cells
>
> **[Chỉ vào nhánh phải - Atomic]**
>
> The **right branch** focuses on **reducing memory footprint**:
> - **Version 4: CfAMe** - changes from Gather to Scatter pattern using atomic operations, but keeps the flow buffer
> - **Version 5: CfAMo** - completely **eliminates the 12MB flow buffer**
>
> As we will see, **CfAMo** turns out to be the **fastest** version."

---

## Slide 5: Time Execution (1 phút)

**[Phân tích kết quả thời gian]**

> "Let's look at the **execution time results** for 16,000 simulation steps.
>
> **[Chỉ vào biểu đồ]**
>
> From the chart, we can see:
> - **CfAMe** is the slowest at **24.69 seconds**
> - **Tiled+Halo** at **23.33 seconds**
> - **Global** baseline at **21.60 seconds**
> - **Tiled** slightly faster at **20.14 seconds**
> - And **CfAMo** is the **fastest** at **19.74 seconds**
>
> **[Chỉ vào bảng Speedup]**
>
> In terms of speedup compared to the baseline:
> - CfAMo achieves **1.09x speedup** - about 9% faster
> - Interestingly, Tiled+Halo is actually **slower** than the baseline
>
> **[Nhấn mạnh Key Finding]**
>
> The key finding here is that **CfAMo wins** by eliminating the 12MB flow buffer, which improves cache efficiency."

---

## Slide 6: Roofline Model (1 phút 15 giây)

**[Phân tích Roofline]**

> "Now let's analyze performance using the **Roofline Model**.
>
> **[Chỉ vào biểu đồ Roofline]**
>
> The roofline plot shows **performance** on the Y-axis versus **Arithmetic Intensity** on the X-axis.
>
> The key observation is that **all five versions** cluster in the **bottom-left region** of the plot, with Arithmetic Intensity **less than 0.05** FLOP per Byte.
>
> **[Chỉ vào bảng AI]**
>
> Looking at the exact values:
> - Global has AI of 0.043
> - Tiled versions around 0.046 to 0.048
> - CfA versions even lower at 0.020 to 0.021
>
> The **ridge point** for GTX 980 is **0.694**. Since all our AI values are far below this threshold, it confirms that **all versions are memory-bound**, not compute-bound.
>
> This low AI is due to the **stencil access pattern** - we need to read 9 neighbors times 3 substates, resulting in many memory accesses per computation."

---

## Slide 7: GPU Occupancy (1 phút)

**[Phân tích Occupancy - Điểm quan trọng]**

> "This slide reveals an **important insight** about GPU occupancy.
>
> **[Chỉ vào biểu đồ Occupancy]**
>
> Looking at the occupancy values:
> - Global has **55%** occupancy
> - Tiled and CfA versions around **58-59%**
> - And **Tiled+Halo** has the **highest** occupancy at **62.7%**
>
> **[Nhấn mạnh insight quan trọng]**
>
> Now here's the **critical insight**: Tiled+Halo has the **highest occupancy**, but it's actually **slower** than the baseline Global version!
>
> This demonstrates that **high occupancy does NOT equal high performance**.
>
> The reason is that the **`__syncthreads()` synchronization overhead** required for shared memory tiling **exceeds the benefits** when the data already fits well in L2 cache.
>
> This is a common misconception in GPU programming that we must be careful about."

---

## Slide 8: FLOP Count (1 phút)

**[Phân tích FLOP và Bottleneck]**

> "Let's examine the **FLOP count** distribution across kernels.
>
> **[Chỉ vào biểu đồ TikZ]**
>
> The chart shows FLOP count per kernel invocation:
> - **massBalance** dominates with **9.56 million FLOPs**
> - **computeOutflows** has only **0.86 million FLOPs**
> - The other kernels have even fewer
>
> **[Chỉ vào bảng Time%]**
>
> But look at the **time distribution**:
> - massBalance takes **27.7%** of GPU time
> - computeOutflows takes **24%** despite having fewer FLOPs
>
> **[Nhấn mạnh Bottleneck]**
>
> Most importantly, **34.7%** of GPU time is spent on **DtoD memory copies** - that's buffer swapping operations!
>
> This confirms our program is **memory-bound** - we spend more time moving data than computing. This is exactly why **CfAMo's elimination of the flow buffer** makes such a significant difference."

---

## Slide 9: Conclusions (1 phút 30 giây)

**[Tổng kết - Nhấn mạnh key points]**

> "Let me summarize the **main conclusions** from this work.
>
> **[Chỉ vào Main Results]**
>
> **First**, CfAMo is the fastest version at 19.74 seconds, achieving 1.09x speedup.
>
> **Second**, all versions are memory-bound, as confirmed by the roofline analysis.
>
> **Third**, and this is important: **high occupancy does not guarantee better performance**.
>
> **Fourth**, reducing memory footprint matters more than compute optimization for this type of workload.
>
> **[Chỉ vào Why CfAMo Wins]**
>
> Why does CfAMo win?
> - It **eliminates the 12MB flow buffer**, reducing memory pressure
> - This leads to **better cache utilization**
> - And since only about **5% of cells have active lava**, atomic contention is minimal
>
> **[Chỉ vào Lesson Learned]**
>
> The **key lesson** is: for small grids that fit in L2 cache, **reducing memory footprint** is more effective than shared memory tiling. Sometimes the simplest optimization - just removing unnecessary data - works better than sophisticated techniques."

---

## Slide 10: Thank You (20 giây)

**[Kết thúc]**

> "That concludes my presentation.
>
> Thank you for your attention. I'm happy to answer any questions you may have.
>
> You can reach me at the email shown on the slide."

---

# Tips for Presentation

## Timing
- **Total: ~10 minutes**
- Slide 1-2: 1 minute
- Slides 3-9: 8 minutes (main content)
- Slide 10: 1 minute (Q&A transition)

## Key Phrases to Emphasize
- "Memory-bound, not compute-bound"
- "High occupancy does NOT equal performance"
- "Reducing memory footprint beats shared memory tiling"
- "34.7% time on memory copies - the real bottleneck"

## Potential Questions & Answers

**Q1: Why not use newer GPU?**
> "GTX 980 was the available hardware. The optimization principles apply to any GPU, though absolute numbers would differ."

**Q2: Would results change for larger grids?**
> "Yes! For grids larger than L2 cache (>2MB), Tiled versions would likely outperform Global. CfAMo should still be competitive due to reduced memory footprint."

**Q3: Why is atomic contention low?**
> "Only ~1.3% of cells have active lava at any time. Most threads skip atomic operations entirely, so collisions are rare."

**Q4: How did you measure execution time?**
> "Using nvprof profiler with 16,000 simulation steps. Times are averaged across multiple runs."

**Q5: What is the accuracy compared to real eruption data?**
> "The focus was on performance, not accuracy. The simulation parameters are calibrated for the 2006 Mt. Etna eruption, but accuracy validation was not part of this work."

---

# Pronunciation Guide (Vietnamese speakers)

| Term | Pronunciation |
|------|---------------|
| Sciara | "shee-AH-rah" |
| Roofline | "ROOF-line" |
| Occupancy | "AH-kyoo-pun-see" |
| Arithmetic Intensity | "air-ith-MEH-tik in-TEN-si-tee" |
| Stencil | "STEN-sil" |
| Throughput | "THROO-put" |
| Bottleneck | "BAH-tl-nek" |
| Cache | "KASH" |
| Halo | "HAY-loh" |

---

**Good luck with your presentation! 🎯**
