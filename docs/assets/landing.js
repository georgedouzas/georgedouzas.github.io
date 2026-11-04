// Landing page effects. Each is optional: without this script all content stays visible.
document.addEventListener("DOMContentLoaded", () => {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  startParticles(reducedMotion);
  followPointerOnCards();
  filterPostsByTag();

  if (!reducedMotion) {
    revealOnScroll();
  }
});

function startParticles(reducedMotion) {
  const hero = document.getElementById("hero-particles");
  if (!hero || typeof window.particlesJS !== "function") {
    return;
  }

  const particleColor = "#8ab4ff";

  window.particlesJS("hero-particles", {
    particles: {
      number: { value: hero.offsetWidth < 700 ? 30 : 80, density: { enable: false } },
      color: { value: particleColor },
      shape: { type: "circle" },
      opacity: { value: 0.6, random: true },
      size: { value: 2.5, random: true },
      line_linked: {
        enable: true,
        distance: 140,
        color: particleColor,
        opacity: 0.3,
        width: 1,
      },
      move: {
        enable: !reducedMotion,
        speed: 1,
        direction: "none",
        random: false,
        straight: false,
        out_mode: "out",
        bounce: false,
      },
    },
    interactivity: {
      detect_on: "canvas",
      events: {
        onhover: { enable: !reducedMotion, mode: "grab" },
        onclick: { enable: false },
        resize: true,
      },
      modes: {
        grab: { distance: 190, line_linked: { opacity: 0.75 } },
      },
    },
    retina_detect: true,
  });
}

// Hides content only once this script is known to be running.
function revealOnScroll() {
  const items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    return;
  }

  document.documentElement.classList.add("reveal-ready");

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    }
  }, { threshold: 0.08 });

  items.forEach((item) => observer.observe(item));
}

function followPointerOnCards() {
  document.querySelectorAll(".oss-card, .quick-card").forEach((card) => {
    card.addEventListener("pointermove", (event) => {
      const box = card.getBoundingClientRect();
      card.style.setProperty("--pointer-x", `${event.clientX - box.left}px`);
      card.style.setProperty("--pointer-y", `${event.clientY - box.top}px`);
    });
  });
}

// Shows only posts with the tag named in the address, such as posts.html#tag=Physics.
function filterPostsByTag() {
  const posts = document.querySelectorAll("article[data-tags]");
  const chips = document.querySelectorAll("[data-tag-filter]");
  if (chips.length === 0) {
    return;
  }

  const apply = () => {
    const match = window.location.hash.match(/^#tag=(.+)$/);
    const tag = match ? decodeURIComponent(match[1]) : "";
    let shownOlder = 0;
    let shown = 0;

    posts.forEach((post) => {
      const visible = !tag || post.dataset.tags.split("|").includes(tag);
      post.hidden = !visible;
      if (visible) {
        shown += 1;
        shownOlder += post.classList.contains("older-post") ? 1 : 0;
      }
    });

    chips.forEach((chip) => {
      const active = chip.dataset.tagFilter === tag;
      chip.classList.toggle("is-active", active);
      if (active) {
        chip.setAttribute("aria-current", "true");
      } else {
        chip.removeAttribute("aria-current");
      }
    });

    const featured = document.querySelector(".featured-post");
    const eyebrow = document.querySelector(".landing-section .landing-eyebrow");
    const olderHeading = document.querySelector(".older-heading");
    const empty = document.querySelector(".tag-empty");
    if (eyebrow) {
      eyebrow.hidden = Boolean(tag) || (featured && featured.hidden);
    }
    if (olderHeading) {
      olderHeading.hidden = Boolean(tag) || shownOlder === 0;
    }
    if (empty) {
      empty.hidden = shown > 0;
    }
  };

  window.addEventListener("hashchange", apply);
  apply();
}
