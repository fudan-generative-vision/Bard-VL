window.SITE_CONFIG = {
  site: {
    title: "BARD-VL",
    subtitle: "Bridging AutoRegressive and Diffusion Vision-Language Models",
  },
  authors: [
    [
      { name: "Baoyou Chen", affiliations: [1, 3], homepage: "https://github.com/cbyzju" },
      { name: "Hanchen Xia", affiliations: [1], homepage: "https://github.com/1ring2rta" },
      { name: "Peng Tu", affiliations: [1], homepage: "https://github.com/yhpengtu-rgb" },
      { name: "Haojun Shi", affiliations: [1], homepage: "https://github.com/Theseus-427" },
    ],
    [
      { name: "Liwei Zhang", affiliations: [1], homepage: "https://github.com/AricGamma" },
      { name: "Weihao Yuan", affiliations: [4], homepage: "https://github.com/weihaosky" },
      { name: "Siyu Zhu", affiliations: [1, 2, 3], marks: ["†"], homepage: "https://sites.google.com/site/zhusiyucs/home" },
    ],
  ],
  affiliations: {
    1: "Shanghai Academy of AI for Science",
    2: "Shanghai Innovation Institute",
    3: "Fudan University",
    4: "Nanjing University",
  },
  links: [
    { label: "Paper", url: "https://arxiv.org/pdf/2604.16514", icon: "pdf", tone: "paper" },
    { label: "arXiv", url: "https://arxiv.org/abs/2604.16514", icon: "arxiv", tone: "outline" },
    { label: "Code", url: "https://github.com/fudan-generative-vision/Bard-VL", icon: "github", tone: "outline" },
    { label: "HuggingFace", url: "https://huggingface.co/collections/fudan-generative-ai/bard-vl", icon: "huggingface", tone: "outline" },
  ],
  hero: {
    media: {
      type: "video",
      src: "assets/qwen3-vs-bard.mp4",
      poster: "assets/receipt-decoding.png",
      autoplay: true,
      controls: false,
    },
    caption:
      "Comparison between autoregressive decoding and block-wise diffusion refinement on long structured multimodal outputs.",
    stats: [
      { value: "5 / 7", label: "benchmarks improved over Qwen3-VL at 4B" },
      { value: "6 / 7", label: "benchmarks improved over Qwen3-VL at 8B" },
      { value: "3x", label: "throughput gain on long outputs" },
    ],
  },
  abstract: {
    title: "Abstract",
    paragraphs: [
      "Autoregressive vision-language models (VLMs) deliver strong multimodal capability, but token-by-token decoding imposes a fundamental inference bottleneck. Diffusion VLMs offer a more parallel decoding paradigm, yet directly converting a pretrained autoregressive VLM into a large-block diffusion VLM often leads to substantial quality degradation.",
      "BARD-VL combines progressive supervised block merging with stage-wise intra-dVLM distillation from a fixed small-block diffusion anchor. It further incorporates mixed-noise training and memory-friendly multimodal packing. With no more than 4.4M training samples, BARD-VL transfers strong multimodal capability from Qwen3-VL to a large-block dVLM while achieving up to 3x decoding throughput speedup.",
    ],
  },
  results: {
    tag: "Results",
    tone: "results",
    title: "Main Results",
    description: "",
    featured: {
      title: "Comparison with autoregressive and diffusion baselines",
      text:
        "Across seven benchmarks, BARD-VL 4B and 8B match or exceed the corresponding Qwen3-VL variants on most reported benchmarks and outperform prior open diffusion VLM baselines in the released comparison.",
      bullets: [
        "BARD-VL-8B outperforms LLaDA-V-8B on all seven reported benchmarks.",
        "BARD-VL-4B outperforms Dimple-VL on all seven reported benchmarks.",
        "The released comparison shows that diffusion decoding need not incur a uniform drop in multimodal performance.",
      ],
      media: {
        type: "image",
        src: "assets/bard-vl-radar.png",
        alt: "Radar chart comparing BARD-VL with autoregressive and diffusion baselines.",
      },
    },
    cards: [
      {
        title: "OCRBench accuracy-throughput trade-off",
        text:
          "On OCRBench, BARD-VL traces a stronger accuracy-throughput frontier across a broad throughput range. At comparable throughput levels, it preserves competitive OCR accuracy while benefiting from substantially faster block-wise decoding. This behavior is especially meaningful for OCR-style tasks with long structured outputs, where autoregressive decoding cost scales directly with output length and the efficiency advantage of diffusion decoding becomes more pronounced.",
        media: {
          type: "image",
          src: "assets/ocrbench-tradeoff.png",
          alt: "OCRBench accuracy-throughput trade-off plot for BARD-VL and baselines.",
        },
      },
      {
        title: "Long structured output with six refinement steps",
        text:
          "On receipt extraction, BARD-VL reaches the final structured output in six refinement steps, whereas Qwen3-VL requires 35 autoregressive decoding steps.",
        media: {
          type: "image",
          src: "assets/receipt-decoding.png",
          alt: "Receipt decoding comparison between Qwen3-VL and BARD-VL.",
        },
      },
    ],
  },
  benchmark: {
    tag: "Benchmark",
    tone: "benchmark",
    title: "Benchmark Comparison",
    description:
      "Comparison with autoregressive and diffusion VLM baselines on seven multimodal benchmarks.",
    columns: [
      { key: "model", label: "Model" },
      { key: "scale", label: "Scale" },
      { key: "mmmu", label: "MMMU<sub>val</sub>" },
      { key: "mmmupro", label: "MMMU-Pro<sub>standard</sub>" },
      { key: "mme", label: "MME<sub>sum</sub>" },
      { key: "rwqa", label: "RealWorldQA" },
      { key: "mmstar", label: "MMStar" },
      { key: "ai2d", label: "AI2D" },
      { key: "chartqa", label: "ChartQA" },
    ],
    rows: [
      { group: "Autoregressive Vision-Language Models" },
      { model: "Qwen3-VL", scale: "4B", mmmu: "47.9", mmmupro: "35.0", mme: "2297", rwqa: "70.5", mmstar: "56.9", ai2d: "81.0", chartqa: "80.9" },
      { model: "Qwen3-VL", scale: "8B", mmmu: "53.0", mmmupro: "36.0", mme: "2379", rwqa: "69.5", mmstar: "59.9", ai2d: "83.5", chartqa: "84.0" },
      { model: "InternVL3.5", scale: "4B", mmmu: "57.4", mmmupro: "38.2", mme: "2236", rwqa: "66.7", mmstar: "65.6", ai2d: "80.6", chartqa: "86.2" },
      { model: "InternVL3.5", scale: "8B", mmmu: "57.2", mmmupro: "41.0", mme: "2359", rwqa: "63.1", mmstar: "66.3", ai2d: "82.1", chartqa: "87.0" },
      { group: "Diffusion Vision-Language Models" },
      { model: "LLaDA-V", scale: "8B", mmmu: "48.8", mmmupro: "35.4", mme: "1998", rwqa: "63.4", mmstar: "60.4", ai2d: "77.8", chartqa: "78.2" },
      { model: "Dream-VL", scale: "7B", mmmu: "51.6", mmmupro: "25.0", mme: "2179", rwqa: "67.7", mmstar: "59.9", ai2d: "80.4", chartqa: "86.2" },
      { model: "LaviDa", scale: "8B", mmmu: "44.2", mmmupro: "28.6", mme: "1711", rwqa: "40.3", mmstar: "47.0", ai2d: "70.1", chartqa: "64.6" },
      { model: "SDAR-VL", scale: "8B", mmmu: "44.0", mmmupro: "28.2", mme: "2142", rwqa: "66.1", mmstar: "53.3", ai2d: "79.6", chartqa: "82.4" },
      { model: "MMaDA", scale: "8B", mmmu: "30.2", mmmupro: "21.5", mme: "1287", rwqa: "28.2", mmstar: "25.7", ai2d: "54.9", chartqa: "43.2" },
      { model: "Dimple-VL", scale: "7B", mmmu: "46.4", mmmupro: "24.1", mme: "1924", rwqa: "51.9", mmstar: "47.7", ai2d: "74.2", chartqa: "58.4" },
      { group: "BARD-VL Converted from Qwen3-VL" },
      { model: "BARD-VL (B=32)", scale: "2B", mmmu: "42.0", mmmupro: "27.9", mme: "2045", rwqa: "64.6", mmstar: "53.1", ai2d: "72.6", chartqa: "76.8", emphasis: true },
      { model: "BARD-VL (B=32)", scale: "4B", mmmu: "53.0", mmmupro: "34.2", mme: "2305", rwqa: "71.9", mmstar: "63.6", ai2d: "82.8", chartqa: "80.2", emphasis: true },
      { model: "BARD-VL (B=4)", scale: "8B", mmmu: "54.6", mmmupro: "37.6", mme: "2393", rwqa: "70.7", mmstar: "65.0", ai2d: "83.2", chartqa: "84.6", emphasis: true },
    ],
    note: "",
  },
  method: {
    tag: "Method",
    tone: "method",
    title: "Method Overview",
    description: "",
    cards: [
      {
        icon: "blocks",
        figure: "progressive-blocks",
        title: "Progressive block merging",
        text:
          "BARD-VL does not transition directly from autoregressive decoding to large-block diffusion. The decoding block size is increased progressively over multiple stages.",
        bullets: [
          "Transitions gradually from causal next-token prediction to block-wise parallel decoding.",
          "Uses intermediate block sizes before reaching the final large-block model.",
          "Reduces degradation during autoregressive-to-diffusion transfer.",
        ],
        schedule: ["B=4", "B=8", "B=16", "B=32"],
      },
      {
        icon: "spark",
        title: "Stage-wise dVLM distillation",
        text:
          "Autoregressive teacher states are not aligned with diffusion denoising states. BARD-VL therefore distills from a diffusion model operating under the same corruption process.",
        bullets: [
          "Aligns teacher and student within the same denoising regime.",
          "Mitigates mismatch between clean autoregressive prefixes and corrupted diffusion states.",
        ],
        media: {
          type: "image",
          src: "assets/method-mismatch.png",
          alt: "Mismatch between autoregressive teacher logits and diffusion student logits.",
        },
      },
      {
        icon: "mask",
        title: "Packed multimodal attention mask",
        text:
          "The packed attention layout shares multimodal context across clean and noisy branches, reducing redundant computation for long multimodal sequences.",
        bullets: [
          "Reuses shared multimodal context instead of duplicating it across branches.",
          "Restricts each noisy block to the multimodal context, the clean prefix, and the current block.",
        ],
        mediaPair: [
          {
            type: "image",
            src: "assets/mask-vanilla.jpg",
            alt: "Vanilla block-wise attention mask.",
            label: "Vanilla",
          },
          {
            type: "image",
            src: "assets/mask-packed.jpg",
            alt: "Packed block-wise attention mask used by BARD.",
            label: "Packed",
          },
        ],
      },
      {
        icon: "noise",
        title: "Mixed-noise training",
        text:
          "Pure absorbing-mask corruption is stable but weaker once incorrect visible tokens appear. BARD-VL combines masked-token and uniform token corruption to support both completion and revision.",
        bullets: [
          "Jointly trains masked-token recovery and visible-token correction.",
          "Improves iterative refinement after tokens become visible.",
        ],
        media: {
          type: "image",
          src: "assets/mixed-noise.jpg",
          alt: "Mixed-noise scheduler coefficients.",
        },
      },
    ],
  },
  demos: {
    tag: "Demos",
    tone: "demos",
    title: "Qualitative Examples",
    description:
      "Examples on structured document decoding.",
    cards: [
      {
        title: "OCRBench document comparison",
        text:
          "Comparison on OCRBench-style document extraction with long structured outputs.",
        layout: "wide",
        media: {
          type: "image",
          src: "assets/ocrbench-comparison.png",
          alt: "OCRBench qualitative comparison between baseline and BARD-VL.",
        },
      },
    ],
  },
  citation: {
    tag: "Citation",
    tone: "citation",
    title: "Citation",
    // description:
    //   "Please cite this work as follows.",
    bibtex: `@misc{chen2026bardbridgingautoregressivediffusion,
      title={BARD: Bridging AutoRegressive and Diffusion Vision-Language Models Via Highly Efficient Progressive Block Merging and Stage-Wise Distillation}, 
      author={Baoyou Chen and Hanchen Xia and Peng Tu and Haojun Shi and Shan Mu and Weihao Yuan and Siyu Zhu},
      year={2026},
      eprint={2604.16514},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.16514}, 
}`,
  },
  footer: {
    note: "",
  },
};
