// Landing page effects. Each is optional: without this script all content stays visible.
document.addEventListener("DOMContentLoaded", () => {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  startParticles(reducedMotion);
  followPointerOnCards();
  filterPostsByTag();
  refreshStarCounts();

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
  document.querySelectorAll(".quick-card").forEach((card) => {
    card.addEventListener("pointermove", (event) => {
      const box = card.getBoundingClientRect();
      card.style.setProperty("--pointer-x", `${event.clientX - box.left}px`);
      card.style.setProperty("--pointer-y", `${event.clientY - box.top}px`);
    });
  });
}

// Updates the star counts built into the Open source page. If GitHub does not answer, those counts stay.
async function refreshStarCounts() {
  const counts = document.querySelectorAll("[data-repo]");
  if (counts.length === 0) {
    return;
  }

  try {
    const response = await fetch("https://api.github.com/users/georgedouzas/repos?per_page=100");
    if (!response.ok) {
      return;
    }
    const repos = new Map((await response.json()).map((repo) => [repo.name, repo.stargazers_count]));
    counts.forEach((count) => {
      if (repos.has(count.dataset.repo)) {
        count.querySelector(".post-row-star-count").textContent = repos.get(count.dataset.repo);
      }
    });
  } catch {
    // Network failures leave the built-in counts in place.
  }
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
    let shown = 0;

    posts.forEach((post) => {
      const visible = !tag || post.dataset.tags.split("|").includes(tag);
      post.hidden = !visible;
      shown += visible ? 1 : 0;
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

    const empty = document.querySelector(".tag-empty");
    if (empty) {
      empty.hidden = shown > 0;
    }
  };

  window.addEventListener("hashchange", apply);
  apply();
}
