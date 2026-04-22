const app = document.getElementById("app");
const config = window.SITE_CONFIG;
let lazyVideoObserver = null;

if (!app || !config) {
  throw new Error("Missing #app or SITE_CONFIG.");
}

const ICONS = {
  pdf: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"></path>
      <path d="M14 3v5h5"></path>
      <path d="M8.5 15.5h1.5a1.5 1.5 0 0 0 0-3H8.5v5"></path>
      <path d="M13 17.5v-5h1.25a1.75 1.75 0 0 1 0 3.5H13"></path>
      <path d="M18 12.5h-2.5v5"></path>
      <path d="M15.5 15h2"></path>
    </svg>
  `,
  arxiv: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4.5 18 9 6l4.5 12"></path>
      <path d="M6.2 13.5h5.6"></path>
      <path d="M14.5 7.5V18"></path>
      <path d="M14.5 12h5"></path>
      <path d="M19.5 7.5V18"></path>
    </svg>
  `,
  github: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M9 19c-4 1.2-4-2-6-2"></path>
      <path d="M15 22v-3.1a3.4 3.4 0 0 0-1-2.6c3.3-.4 6.8-1.6 6.8-7.1A5.5 5.5 0 0 0 19.2 5 5.1 5.1 0 0 0 19.1 1.5S17.8 1.1 15 3a13 13 0 0 0-6 0C6.2 1.1 4.9 1.5 4.9 1.5A5.1 5.1 0 0 0 4.8 5a5.5 5.5 0 0 0-1.6 4.2c0 5.5 3.5 6.7 6.8 7.1a3.4 3.4 0 0 0-1 2.6V22"></path>
    </svg>
  `,
  huggingface: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="9" cy="10" r="1.2"></circle>
      <circle cx="15" cy="10" r="1.2"></circle>
      <path d="M7 14.2c1.1 1 2.7 1.6 5 1.6s3.9-.6 5-1.6"></path>
      <path d="M6.5 11.5c-.9.6-1.5 1.7-1.5 2.9 0 2.4 2.2 4.6 7 4.6s7-2.2 7-4.6c0-1.2-.6-2.3-1.5-2.9"></path>
      <path d="M7.2 8.4c0-2.1 1.7-3.9 4.8-3.9s4.8 1.8 4.8 3.9"></path>
    </svg>
  `,
  chart: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 19h16"></path>
      <path d="M7 16V9"></path>
      <path d="M12 16V5"></path>
      <path d="M17 16v-4"></path>
    </svg>
  `,
  layers: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m12 4 8 4-8 4-8-4 8-4Z"></path>
      <path d="m4 12 8 4 8-4"></path>
      <path d="m4 16 8 4 8-4"></path>
    </svg>
  `,
  quote: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M9 11H6.5A2.5 2.5 0 0 1 9 8.5V7A4 4 0 0 0 5 11v6h6v-6Z"></path>
      <path d="M19 11h-2.5A2.5 2.5 0 0 1 19 8.5V7a4 4 0 0 0-4 4v6h6v-6Z"></path>
    </svg>
  `,
  blocks: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="4" y="4" width="6" height="6" rx="1.5"></rect>
      <rect x="14" y="4" width="6" height="6" rx="1.5"></rect>
      <rect x="4" y="14" width="6" height="6" rx="1.5"></rect>
      <path d="M14 17h6"></path>
      <path d="M17 14v6"></path>
    </svg>
  `,
  spark: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z"></path>
      <path d="M19 16.5 20 19l2.5 1-2.5 1L19 23l-1-2-2.5-1 2.5-1 1-2.5Z"></path>
    </svg>
  `,
  mask: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="4" y="5" width="16" height="14" rx="2"></rect>
      <path d="M4 10h16"></path>
      <path d="M9 5v14"></path>
      <path d="M15 10v9"></path>
    </svg>
  `,
  noise: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 12c2.5 0 2.5-6 5-6s2.5 12 5 12 2.5-12 5-12 2.5 6 5 6"></path>
    </svg>
  `,
};

const isInternalLink = (url) => typeof url === "string" && url.startsWith("#");
const RESOURCE_ICON_CLASSES = {
  pdf: "icon-lm-pdf",
  arxiv: "icon-lm-Arxiv",
  github: "icon-lm-github",
  huggingface: "icon-lm-huggingface",
};

function iconSvg(name, className = "icon") {
  if (!name || !ICONS[name]) return "";
  return `<span class="${className}" aria-hidden="true">${ICONS[name]}</span>`;
}

function renderButtonIcon(name, className = "button__icon") {
  const iconClass = RESOURCE_ICON_CLASSES[name];
  if (iconClass) {
    return `<i class="iconfont ${iconClass} ${className}" aria-hidden="true"></i>`;
  }
  return iconSvg(name, className);
}

function mediaType(media) {
  if (!media) return "image";
  if (media.type) return media.type;
  if (!media.src) return "image";
  const ext = media.src.split("?")[0].split("#")[0].split(".").pop().toLowerCase();
  return ["mp4", "mov", "webm"].includes(ext) ? "video" : "image";
}

function renderMedia(media, options = {}) {
  if (!media) return "";

  const type = mediaType(media);
  const classes = ["media-frame"];
  if (options.className) classes.push(options.className);
  if (options.contain) classes.push("media-frame--contain");

  if (type === "video") {
    const autoplay = options.autoplay ?? media.autoplay ?? false;
    const controls = options.controls ?? media.controls ?? true;
    const lazy = options.lazy ?? media.lazy ?? !autoplay;
    const sourceAttr = lazy ? `data-src="${media.src}"` : `src="${media.src}"`;
    const videoClass = lazy ? 'class="js-lazy-video"' : "";
    return `
      <div class="${classes.join(" ")}">
        <video
          ${videoClass}
          ${sourceAttr}
          ${media.poster ? `poster="${media.poster}"` : ""}
          ${autoplay ? "autoplay muted loop playsinline" : "playsinline"}
          ${controls ? "controls" : ""}
          preload="${lazy ? "none" : media.preload || "metadata"}"
        ></video>
      </div>
    `;
  }

  return `
    <div class="${classes.join(" ")}">
      <img
        src="${media.src}"
        alt="${media.alt || media.title || "Project asset"}"
        loading="${options.eager ? "eager" : "lazy"}"
      />
    </div>
  `;
}

function renderButton(link) {
  const attrs = isInternalLink(link.url) ? "" : 'target="_blank" rel="noreferrer"';
  return `
    <a class="button button--${link.tone || "default"}" href="${link.url}" ${attrs}>
      ${renderButtonIcon(link.icon, "button__icon")}
      <span>${link.label}</span>
    </a>
  `;
}

function renderAuthor(author) {
  const prefix = author.prefix ? `<sup class="author-mark author-mark--prefix">${author.prefix}</sup>` : "";
  const suffixParts = [];
  if (author.affiliations?.length) {
    suffixParts.push(author.affiliations.join(","));
  }
  if (author.marks?.length) {
    suffixParts.push(...author.marks);
  }
  const suffix = author.suffix || suffixParts.join(",");
  const name = author.homepage
    ? `<a class="author-name" href="${author.homepage}" target="_blank" rel="noreferrer">${author.name}</a>`
    : `<span class="author-name">${author.name}</span>`;

  return `<span class="author-entry">${prefix}${name}${suffix ? `<sup class="author-mark author-mark--suffix">${suffix}</sup>` : ""}</span>`;
}

function renderAuthors() {
  return config.authors
    .map(
      (row, index) => `
        <div class="author-row ${index === config.authors.length - 1 ? "author-row--thin" : ""}">
          ${row.map(renderAuthor).join("")}
        </div>
      `,
    )
    .join("");
}

function renderAffiliations() {
  const institutions = Object.entries(config.affiliations)
    .sort(([left], [right]) => Number(left) - Number(right))
    .map(
      ([key, value]) =>
        `<span class="hero__affiliation"><span class="hero__affiliation-index">${key}</span><span>${value}</span></span>`,
    )
    .join("");

  const notes = (config.authorNotes || [])
    .map((note) => `<span class="hero__affiliation-note">${note}</span>`)
    .join("");
  return `<div class="hero__affiliation-list">${institutions}</div>${notes ? `<div class="hero__affiliation-notes">${notes}</div>` : ""}`;
}

function renderSectionHeading(section) {
  return `
    <div class="section-heading">
      ${section.tag ? `<div class="section-tag section-tag--${section.tone || "default"}">${section.tag}</div>` : ""}
      <h2>${section.title}</h2>
      ${section.description ? `<p>${section.description}</p>` : ""}
    </div>
  `;
}

function renderHero() {
  return `
    <section class="hero section" id="top">
      <div class="container">
        <h1>${config.site.title}</h1>
        <p class="hero__subtitle">${config.site.subtitle}</p>
        ${config.site.description ? `<p class="hero__description">${config.site.description}</p>` : ""}
        <div class="hero__authors">${renderAuthors()}</div>
        <div class="hero__affiliations">${renderAffiliations()}</div>
        <div class="button-row">${config.links.map(renderButton).join("")}</div>
        <div class="hero__media">
          ${renderMedia(config.hero.media, { autoplay: true, controls: false, eager: true, lazy: false })}
          <p class="media-caption">${config.hero.caption}</p>
        </div>
      </div>
    </section>
  `;
}

function renderAbstract() {
  return `
    <section class="section section--narrow" id="abstract">
      <div class="container container--narrow">
        ${renderSectionHeading(config.abstract)}
        <div class="abstract-copy">
          ${config.abstract.paragraphs.map((paragraph) => `<p>${paragraph}</p>`).join("")}
        </div>
      </div>
    </section>
  `;
}

function renderResults() {
  return `
    <section class="section section--results" id="results">
      <div class="container">
        ${renderSectionHeading(config.results)}
        <article class="panel panel--split">
          <div class="panel__copy">
            <h3>${config.results.featured.title}</h3>
            <p>${config.results.featured.text}</p>
            <ul class="detail-list">
              ${config.results.featured.bullets.map((item) => `<li>${item}</li>`).join("")}
            </ul>
          </div>
          ${renderMedia(config.results.featured.media, { contain: true })}
        </article>
        <div class="card-grid">
          ${config.results.cards
            .map((card, index) => renderStoryCard(card, { reverse: index % 2 === 1, contain: true }))
            .join("")}
        </div>
      </div>
    </section>
  `;
}

function renderBenchmark() {
  const columns = config.benchmark.columns;
  const rows = config.benchmark.rows
    .map((row) => {
      if (row.group) {
        return `
          <tr class="table-group">
            <th colspan="${columns.length}">
              <span class="table-group__label">${row.group}</span>
            </th>
          </tr>
        `;
      }

      return `
        <tr class="${row.emphasis ? "table-row--emphasis" : ""}">
          ${columns.map((column) => `<td>${row[column.key] || ""}</td>`).join("")}
        </tr>
      `;
    })
    .join("");

  return `
    <section class="section section--benchmark" id="benchmark">
      <div class="container">
        ${renderSectionHeading(config.benchmark)}
        <div class="table-panel">
          <div class="table-wrap">
            <table class="benchmark-table">
              <thead>
                <tr>${columns.map((column) => `<th>${column.label}</th>`).join("")}</tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>
        ${config.benchmark.note ? `<p class="section-note">${config.benchmark.note}</p>` : ""}
      </div>
    </section>
  `;
}

function parseBlockSize(step) {
  const match = String(step).match(/(\d+)/);
  return match ? Number(match[1]) : null;
}

function renderProgressiveBlocks(card) {
  const sizes = (card.schedule || []).map(parseBlockSize).filter(Boolean);
  if (!sizes.length) return "";

  const total = Math.max(...sizes);

  return `
    <div class="block-progress" aria-label="Progressive block schedule">
      <div class="block-progress__title">Progressive block schedule</div>
      <div class="block-progress__rows">
        ${sizes
          .map((size) => {
            const groups = total / size;
            const segmentWidth = `${(size / total) * 100}%`;
            return `
              <div class="block-progress__row">
                <div class="block-progress__label">B=${size}</div>
                <div class="block-progress__track">
                  ${Array.from({ length: groups }, () => `<span class="block-progress__segment" style="width:${segmentWidth}"></span>`).join("")}
                </div>
              </div>
            `;
          })
          .join("")}
      </div>
      <p class="block-progress__caption">BARD-VL increases block size over multiple training stages before reaching large-block decoding.</p>
    </div>
  `;
}

function renderMethodCard(card) {
  const figure = card.figure === "progressive-blocks" ? renderProgressiveBlocks(card) : "";

  const schedule = card.schedule && !figure
    ? `
      <div class="schedule-strip">
        ${card.schedule.map((step) => `<span>${step}</span>`).join("")}
      </div>
    `
    : "";

  const mediaPair = card.mediaPair
    ? `
      <div class="pair-grid">
        ${card.mediaPair
          .map(
            (item) => `
              <div class="pair-grid__item">
                ${renderMedia(item, { contain: true })}
                <div class="pair-grid__label">${item.label}</div>
              </div>
            `,
          )
          .join("")}
      </div>
    `
    : "";

  return `
    <article class="panel method-card">
      <div class="panel__copy">
        <h3>${card.title}</h3>
        <p>${card.text}</p>
        ${card.bullets ? `<ul class="detail-list">${card.bullets.map((item) => `<li>${item}</li>`).join("")}</ul>` : ""}
      </div>
      ${figure}
      ${schedule}
      ${card.media ? renderMedia(card.media, { contain: true }) : ""}
      ${mediaPair}
    </article>
  `;
}

function renderStoryCard(card, options = {}) {
  const reverse = options.reverse ? " panel--reverse" : "";

  if (card.layout === "wide") {
    return `
      <article class="panel panel--stack panel--wide">
        <div class="panel__copy">
          <h3>${card.title}</h3>
          <p>${card.text}</p>
        </div>
        ${renderMedia(card.media, { controls: options.controls, contain: options.contain })}
      </article>
    `;
  }

  return `
    <article class="panel panel--story${reverse}">
      <div class="panel__media">
        ${renderMedia(card.media, { controls: options.controls, contain: options.contain })}
      </div>
      <div class="panel__copy">
        <h3>${card.title}</h3>
        <p>${card.text}</p>
        ${card.caption ? `<p class="panel__caption">${card.caption}</p>` : ""}
      </div>
    </article>
  `;
}

function hydrateLazyVideo(video) {
  if (!video || !video.dataset.src) return;
  video.src = video.dataset.src;
  video.removeAttribute("data-src");
  video.load();
}

function initLazyVideos() {
  if (lazyVideoObserver) {
    lazyVideoObserver.disconnect();
    lazyVideoObserver = null;
  }

  const videos = Array.from(app.querySelectorAll("video[data-src]"));
  if (!videos.length) return;

  if (!("IntersectionObserver" in window)) {
    videos.forEach(hydrateLazyVideo);
    return;
  }

  lazyVideoObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        hydrateLazyVideo(entry.target);
        lazyVideoObserver.unobserve(entry.target);
      });
    },
    {
      rootMargin: "240px 0px",
      threshold: 0.01,
    },
  );

  videos.forEach((video) => lazyVideoObserver.observe(video));
}

function renderMethod() {
  return `
    <section class="section section--method" id="method">
      <div class="container">
        ${renderSectionHeading(config.method)}
        <div class="card-grid card-grid--two">
          ${config.method.cards.map(renderMethodCard).join("")}
        </div>
      </div>
    </section>
  `;
}

function renderDemos() {
  return `
    <section class="section section--demos" id="demos">
      <div class="container">
        ${renderSectionHeading(config.demos)}
        <div class="card-grid">
          ${config.demos.cards
            .map((card, index) =>
              renderStoryCard(card, {
                reverse: index % 2 === 1,
                controls: true,
                contain: mediaType(card.media) !== "video",
              }),
            )
            .join("")}
        </div>
      </div>
    </section>
  `;
}

function renderCitation() {
  return `
    <section class="section section--citation" id="citation">
      <div class="container container--narrow">
        <div class="citation">
          <div class="citation__header">
            <div>
              <h2>${config.citation.title}</h2>
              ${config.citation.description ? `<p>${config.citation.description}</p>` : ""}
            </div>
            <button class="button button--subtle button--copy" type="button" data-copy-bibtex><span>Copy</span></button>
          </div>
          <pre class="citation-block">${config.citation.bibtex}</pre>
        </div>
      </div>
    </section>
  `;
}

function renderFooter() {
  if (!config.footer || !config.footer.note) {
    return "";
  }

  return `
    <footer class="site-footer">
      <div class="container container--narrow">
        <p>${config.footer.note}</p>
      </div>
    </footer>
  `;
}

function mount() {
  document.title = config.site.subtitle
    ? `${config.site.title}: ${config.site.subtitle}`
    : config.site.title;

  const meta = document.querySelector('meta[name="description"]');
  const description = config.site.description || config.abstract?.paragraphs?.[0];
  if (meta && description) {
    meta.setAttribute("content", description);
  }

  app.innerHTML = `
    <main class="page">
      ${renderHero()}
      ${renderAbstract()}
      ${renderResults()}
      ${renderBenchmark()}
      ${renderMethod()}
      ${renderCitation()}
      ${renderFooter()}
    </main>
  `;

  initLazyVideos();

  const copyButton = app.querySelector("[data-copy-bibtex]");
  if (copyButton) {
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(config.citation.bibtex);
        copyButton.querySelector("span").textContent = "Copied";
      } catch (error) {
        copyButton.querySelector("span").textContent = "Copy failed";
      }

      window.setTimeout(() => {
        copyButton.querySelector("span").textContent = "Copy";
      }, 1400);
    });
  }
}

mount();
