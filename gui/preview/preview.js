/* TouchSign UI concept — lightweight preview interactions only.
   This file is intentionally standalone; production logic will live in NiceGUI. */

(() => {
  "use strict";

  const zodiac = {
    rat: {
      glyph: "鼠", title: "子鼠", pinyin: "ZI", english: "RAT", branch: "子",
      index: "01", element: "水", direction: "北", directionEn: "NORTH",
      time: "子时", nature: "属水 · 居北", confidence: 98.7,
      poem: "灵鼠衔金，机巧通灵"
    },
    ox: {
      glyph: "牛", title: "丑牛", pinyin: "CHOU", english: "OX", branch: "丑",
      index: "02", element: "土", direction: "东北", directionEn: "NORTHEAST",
      time: "丑时", nature: "属土 · 居东北", confidence: 97.9,
      poem: "厚德载物，耕耘有时"
    },
    tiger: {
      glyph: "虎", title: "寅虎", pinyin: "YIN", english: "TIGER", branch: "寅",
      index: "03", element: "木", direction: "东北", directionEn: "NORTHEAST",
      time: "寅时", nature: "属木 · 居东北", confidence: 98.1,
      poem: "风从虎啸，威镇山林"
    },
    rabbit: {
      glyph: "兔", title: "卯兔", pinyin: "MAO", english: "RABBIT", branch: "卯",
      index: "04", element: "木", direction: "东", directionEn: "EAST",
      time: "卯时", nature: "属木 · 居东", confidence: 97.6,
      poem: "月华凝露，静中藏慧"
    },
    dragon: {
      glyph: "龙", title: "辰龙", pinyin: "CHEN", english: "DRAGON", branch: "辰",
      index: "05", element: "土", direction: "东南", directionEn: "SOUTHEAST",
      time: "辰时", nature: "属土 · 居东南", confidence: 96.8,
      poem: "云起龙骧，万象更新"
    },
    snake: {
      glyph: "蛇", title: "巳蛇", pinyin: "SI", english: "SNAKE", branch: "巳",
      index: "06", element: "火", direction: "东南", directionEn: "SOUTHEAST",
      time: "巳时", nature: "属火 · 居东南", confidence: 96.4,
      poem: "灵蛇盘玉，幽微见真"
    },
    horse: {
      glyph: "马", title: "午马", pinyin: "WU", english: "HORSE", branch: "午",
      index: "07", element: "火", direction: "南", directionEn: "SOUTH",
      time: "午时", nature: "属火 · 居南", confidence: 97.2,
      poem: "天马行空，一往无前"
    },
    goat: {
      glyph: "羊", title: "未羊", pinyin: "WEI", english: "GOAT", branch: "未",
      index: "08", element: "土", direction: "西南", directionEn: "SOUTHWEST",
      time: "未时", nature: "属土 · 居西南", confidence: 95.9,
      poem: "温润如玉，和合生祥"
    },
    monkey: {
      glyph: "猴", title: "申猴", pinyin: "SHEN", english: "MONKEY", branch: "申",
      index: "09", element: "金", direction: "西南", directionEn: "SOUTHWEST",
      time: "申时", nature: "属金 · 居西南", confidence: 96.7,
      poem: "灵猴献瑞，机变无穷"
    },
    rooster: {
      glyph: "鸡", title: "酉鸡", pinyin: "YOU", english: "ROOSTER", branch: "酉",
      index: "10", element: "金", direction: "西", directionEn: "WEST",
      time: "酉时", nature: "属金 · 居西", confidence: 97.1,
      poem: "金鸡报晓，光启东方"
    },
    dog: {
      glyph: "狗", title: "戌狗", pinyin: "XU", english: "DOG", branch: "戌",
      index: "11", element: "土", direction: "西北", directionEn: "NORTHWEST",
      time: "戌时", nature: "属土 · 居西北", confidence: 96.3,
      poem: "忠守其门，信义相随"
    },
    pig: {
      glyph: "猪", title: "亥猪", pinyin: "HAI", english: "PIG", branch: "亥",
      index: "12", element: "水", direction: "西北", directionEn: "NORTHWEST",
      time: "亥时", nature: "属水 · 居西北", confidence: 95.7,
      poem: "亥水生财，丰饶安泰"
    }
  };

  const zodiacOrder = [
    "rat", "ox", "tiger", "rabbit", "dragon", "snake",
    "horse", "goat", "monkey", "rooster", "dog", "pig"
  ];

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const els = {
    cards: $$(".zodiac-card"),
    resultSeal: $(".result-seal"),
    resultGlyph: $("#result-glyph"),
    resultTitle: $("#result-title"),
    resultPinyin: $("#result-pinyin"),
    resultEnglish: $("#result-english"),
    resultNature: $("#result-nature"),
    resultIndex: $("#result-index"),
    resultBranch: $("#result-branch"),
    resultPoem: $("#result-poem"),
    oracleElement: $("#oracle-element"),
    oracleTime: $("#oracle-time"),
    confidenceValue: $("#confidence-value"),
    confidenceBar: $("#confidence-bar"),
    confidenceFill: $("#confidence-fill"),
    sequenceCurrent: $("#sequence-current"),
    candidateNames: [
      $("#candidate-name-1"),
      $("#candidate-name-2"),
      $("#candidate-name-3")
    ],
    candidateBars: [
      $("#candidate-bar-1"),
      $("#candidate-bar-2"),
      $("#candidate-bar-3")
    ],
    candidateValues: [
      $("#candidate-value-1"),
      $("#candidate-value-2"),
      $("#candidate-value-3")
    ],
    systemStatus: $("#system-status"),
    clock: $("#clock"),
    date: $("#date"),
    sensorSelect: $("#sensor-select"),
    cameraMeta: $("#camera-meta"),
    refreshButton: $("#refresh-button"),
    connectButton: $("#connect-button"),
    connectLabel: $("#connect-label"),
    app: $(".app")
  };

  let connected = true;
  let busy = false;
  let statusTimer = 0;

  const setStatus = (text) => {
    els.systemStatus.textContent = text;
  };

  const temporaryStatus = (text, resetAfter = 1200) => {
    window.clearTimeout(statusTimer);
    setStatus(text);
    statusTimer = window.setTimeout(() => {
      setStatus(connected ? "传感器在线" : "传感器离线");
    }, resetAfter);
  };

  const updateClock = () => {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    els.clock.textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    els.date.textContent = `${now.getFullYear()}.${pad(now.getMonth() + 1)}.${pad(now.getDate())}`;
  };

  const animateResult = () => {
    if (reducedMotion) return;

    els.resultSeal.animate(
      [
        { transform: "scale(0.94)", opacity: 0.55, filter: "brightness(1.6)" },
        { transform: "scale(1.025)", opacity: 1, filter: "brightness(1)" },
        { transform: "scale(1)", opacity: 1 }
      ],
      { duration: 480, easing: "cubic-bezier(.22,1,.36,1)" }
    );

    els.resultTitle.animate(
      [
        { opacity: 0.3, transform: "translateY(4px)" },
        { opacity: 1, transform: "translateY(0)" }
      ],
      { duration: 360, easing: "ease-out" }
    );
  };

  const updateCandidates = (key) => {
    const selectedIndex = zodiacOrder.indexOf(key);
    const others = [1, 2].map((offset) => zodiacOrder[(selectedIndex + offset) % zodiacOrder.length]);
    const residual = Math.max(0.1, 100 - zodiac[key].confidence);
    const values = [
      zodiac[key].confidence,
      Math.max(0.2, residual * 0.48),
      Math.max(0.1, residual * 0.2)
    ];
    const entries = [key, ...others];

    entries.forEach((entryKey, i) => {
      const entry = zodiac[entryKey];
      els.candidateNames[i].innerHTML = `${entry.glyph} <small>${entry.title}</small>`;
      els.candidateBars[i].style.width = `${values[i]}%`;
      els.candidateValues[i].innerHTML = `${values[i].toFixed(1)}<em>%</em>`;
    });
  };

  const selectZodiac = (key, { animate = true } = {}) => {
    const data = zodiac[key];
    if (!data) return;

    els.cards.forEach((card) => {
      const active = card.dataset.zodiac === key;
      card.classList.toggle("is-active", active);
      card.setAttribute("aria-pressed", String(active));
    });

    els.resultGlyph.textContent = data.glyph;
    els.resultTitle.textContent = data.title;
    els.resultPinyin.textContent = data.pinyin;
    els.resultEnglish.textContent = data.english;
    els.resultNature.textContent = data.nature;
    els.resultIndex.textContent = data.index;
    els.resultBranch.textContent = data.branch;
    els.resultPoem.textContent = data.poem;
    els.sequenceCurrent.textContent = data.index;
    els.oracleElement.innerHTML = `${data.directionEn} <i>·</i> ${data.element}`;
    els.oracleTime.innerHTML = `${data.time} <i>·</i> ${data.index}`;
    els.confidenceValue.innerHTML = `${data.confidence.toFixed(1)}<em>%</em>`;
    els.confidenceBar.setAttribute("aria-valuenow", String(data.confidence));
    els.confidenceFill.style.width = `${data.confidence}%`;
    els.confidenceFill.style.setProperty("--fill", `${data.confidence}%`);
    updateCandidates(key);

    if (animate) animateResult();
  };

  els.cards.forEach((card) => {
    card.addEventListener("click", () => {
      selectZodiac(card.dataset.zodiac);
      temporaryStatus(`${zodiac[card.dataset.zodiac].title} · 预览已更新`, 1000);
    });
  });

  els.connectButton.addEventListener("click", () => {
    if (busy) return;
    busy = true;
    els.connectButton.classList.add("is-connecting");
    els.connectButton.classList.remove("is-disconnected");
    els.connectLabel.textContent = "正在连接";
    setStatus("设备握手中");

    window.setTimeout(() => {
      connected = true;
      busy = false;
      els.connectButton.classList.remove("is-connecting");
      els.connectButton.classList.remove("is-disconnected");
      els.connectLabel.textContent = "连接传感器";
      els.app.classList.remove("is-sensor-offline");
      setStatus("传感器在线");
    }, 900);
  });

  els.sensorSelect.addEventListener("change", () => {
    const option = els.sensorSelect.selectedOptions[0];
    const cameraCode = els.sensorSelect.value;
    const resolution = option?.dataset.resolution || "640 × 480";
    const code = cameraCode.split("-").pop() || "01";
    els.cameraMeta.innerHTML = `CAM ${code} <b>/</b> ${resolution}`;
    temporaryStatus("正在切换设备", 700);
    if (els.connectLabel.textContent === "断开连接") {
      els.connectLabel.textContent = "连接传感器";
    }
  });

  els.refreshButton.addEventListener("click", () => {
    if (busy) return;
    els.refreshButton.classList.add("is-spinning");
    temporaryStatus("扫描传感设备", 900);
    window.setTimeout(() => els.refreshButton.classList.remove("is-spinning"), 750);
  });

  updateClock();
  window.setInterval(updateClock, 1000);
  selectZodiac("rat", { animate: false });
})();
