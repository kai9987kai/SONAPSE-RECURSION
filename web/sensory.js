// Episodic visual prediction on a fixed recurrent graph. All media stays in the browser.

(() => {
  const $ = (id) => document.getElementById(id);

  const stimuli = SENSORY_STIMULI_DATA;

  const video = $("senseVideo"),
    status = $("senseStatus");

  const W = 20,
    H = 12,
    N = 18,
    LIMIT = 240,
    KEY = "sonapse.visual.episodes.v2";

  const capture = document.createElement("canvas");

  capture.width = W;
  capture.height = H;

  const reader = capture.getContext("2d", { willReadFrequently: true });

  const output = $("dream").getContext("2d"),
    graph = $("senseGraph").getContext("2d");

  const edges = Array.from({ length: N }, (_, i) => [
    [(i + 1) % N, 0.48],
    [(i + 5) % N, -0.29],
    [(i + 11) % N, 0.21],
  ]);

  let memory = [],
    state = Array(N).fill(0),
    last = null,
    dream = null;

  let token = 0,
    interval = 0,
    url = null,
    epoch = "",
    name = "",
    tick = 0,
    lastTime = -1;

  let predictionError = 0,
    holdError = 0,
    scored = 0;

  let episodeTick = 0,
    drive = 0,
    hostState = [0, 0, 0],
    mode = "Idle";

  function check(entry) {
    return (
      entry &&
      typeof entry.epoch === "string" &&
      typeof entry.source === "string" &&
      Number.isSafeInteger(entry.tick) &&
      Array.isArray(entry.pixels) &&
      entry.pixels.length === W * H * 3 &&
      entry.pixels.every((x) => Number.isInteger(x) && x >= 0 && x <= 255) &&
      Array.isArray(entry.state) &&
      Array.isArray(entry.host) &&
      entry.host.length === 3 &&
      entry.host.every(Number.isFinite) &&
      entry.state.length === N &&
      entry.state.every((x) => Number.isFinite(x) && Math.abs(x) <= 1)
    );
  }

  try {
    const stored = JSON.parse(localStorage.getItem(KEY) || "[]");

    if (Array.isArray(stored) && stored.length <= LIMIT && stored.every(check))
      memory = stored;
  } catch {
    status.textContent =
      "Local memory is unavailable. Session learning is available.";
  }

  function advance(nodes, pixels) {
    const next = nodes.map((v, i) => {
      const rx = i % 6,
        ry = Math.floor(i / 6);

      let luminance = 0,
        count = 0;

      for (let y = ry * 4; y < ry * 4 + 4; y++)
        for (
          let x = Math.floor((rx * W) / 6);
          x < Math.floor(((rx + 1) * W) / 6);
          x++
        ) {
          const k = (y * W + x) * 3;

          luminance +=
            (pixels[k] * 0.2126 +
              pixels[k + 1] * 0.7152 +
              pixels[k + 2] * 0.0722) /
            255;
          count++;
        }

      return 0.4 * v + 1.4 * (luminance / count - 0.5);
    });

    edges.forEach((out, i) =>
      out.forEach(([j, w]) => {
        next[j] += nodes[i] * w;
      }),
    );

    return next.map((v) => Math.tanh(v));
  }

  function forecast(frame) {
    let best = Infinity,
      match = -1;

    for (let i = 0; i + 1 < memory.length; i++) {
      const a = memory[i],
        b = memory[i + 1];

      if (a.epoch !== b.epoch || b.tick !== a.tick + 1) continue;

      let distance = 0;

      for (let j = 0; j < a.pixels.length; j += 3) {
        distance +=
          (Math.abs(frame.pixels[j] - a.pixels[j]) +
            Math.abs(frame.pixels[j + 1] - a.pixels[j + 1]) +
            Math.abs(frame.pixels[j + 2] - a.pixels[j + 2])) /
          (255 * 3 * W * H);
      }

      for (let j = 0; j < N; j++)
        distance += (0.4 * Math.abs(frame.state[j] - a.state[j])) / N;

      for (let j = 0; j < 3; j++)
        distance += (0.12 * Math.abs(frame.host[j] - a.host[j])) / 3;

      if (distance < best) {
        best = distance;
        match = i;
      }
    }

    if (match < 0) return null;

    const a = memory[match],
      b = memory[match + 1];

    const pixels = frame.pixels.map((v, i) =>
      Math.round(
        Math.max(
          0,
          Math.min(
            255,
            0.25 * b.pixels[i] + 0.75 * (v + b.pixels[i] - a.pixels[i]),
          ),
        ),
      ),
    );

    const next = advance(frame.state, pixels);

    return {
      pixels,
      state: next,
      host: neural(
        frame.host,
        next.reduce((a, b) => a + b, 0) / N,
        models.circuit,
      ),
      source: b.source,
    };
  }

  function draw(frame) {
    if (!frame) {
      output.clearRect(0, 0, W, H);
      return;
    }

    const image = output.createImageData(W, H);

    for (let i = 0; i < W * H; i++) {
      for (let c = 0; c < 3; c++)
        image.data[i * 4 + c] = frame.pixels[i * 3 + c];
      image.data[i * 4 + 3] = 255;
    }

    output.putImageData(image, 0, 0);
  }

  function render() {
    const chooser = $("recallEpisode"),
      groups = new Map(memory.map((f) => [f.epoch, f.source]));
    if (
      chooser.length !== groups.size + 1 ||
      [...groups.keys()].some((k, i) => chooser.options[i + 1]?.value !== k)
    ) {
      const selection = chooser.value;
      chooser.replaceChildren(
        new Option("Latest observation", ""),
        ...[...groups].map(
          ([key, src], i) => new Option(i + 1 + ". " + src, key),
        ),
      );
      if (groups.has(selection)) chooser.value = selection;
    }

    const points = state.map((_, i) => [
      27 + (i % 6) * 53,
      42 + Math.floor(i / 6) * 73,
    ]);

    graph.fillStyle = "#11252a";
    graph.fillRect(0, 0, 400, 240);

    edges.forEach((out, i) =>
      out.forEach(([j, w]) => {
        const [x, y] = points[i],
          [xx, yy] = points[j];

        graph.strokeStyle = w > 0 ? "#6be4bc" : "#ffa86d";
        graph.globalAlpha = 0.12 + 0.75 * Math.abs(state[i] * w);

        graph.beginPath();
        graph.moveTo(x, y);
        graph.lineTo(xx, yy);
        graph.stroke();

        graph.fillStyle = graph.strokeStyle;
        graph.beginPath();
        graph.arc(0.22 * x + 0.78 * xx, 0.22 * y + 0.78 * yy, 2, 0, 7);
        graph.fill();
      }),
    );

    graph.globalAlpha = 1;

    points.forEach(([x, y], i) => {
      graph.fillStyle = state[i] >= 0 ? "#8eeab4" : "#ffb585";
      graph.beginPath();
      graph.arc(x, y, 3 + 10 * Math.abs(state[i]), 0, 7);
      graph.fill();
    });

    hostState.forEach((v, i) => {
      graph.fillStyle = v >= 0 ? "#a6dfdd" : "#ffa86d";
      graph.fillRect(343, 54 + i * 60, 5 + 40 * Math.abs(v), 9);
    });

    graph.fillStyle = "#d6e9df";
    graph.font = "12px sans-serif";
    graph.fillText("Host", 345, 33);

    $("senseLog").textContent =
      mode +
      " · " +
      tick +
      " updates · " +
      memory.length +
      "/" +
      LIMIT +
      " memory frames" +
      (scored
        ? " · Forecast / hold MAE " +
          (predictionError / scored).toFixed(3) +
          " / " +
          (holdError / scored).toFixed(3)
        : "") +
      (dream ? " · recalled context: " + dream.source : "");
  }

  function halt() {
    token++;
    clearInterval(interval);
    interval = 0;
    video.pause();

    if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());

    video.srcObject = null;
    video.removeAttribute("src");
    video.load();

    if (url) {
      URL.revokeObjectURL(url);
      url = null;
    }

    mode = "Idle";
    render();
  }

  function inputStep() {
    if (video.readyState < 2 || lastTime === video.currentTime) return;

    if (video.currentTime < lastTime) {
      last = null;
      dream = null;
      epoch = crypto.randomUUID();
      episodeTick = 0;
      state = Array(N).fill(0);
      hostState = [0, 0, 0];
    }

    lastTime = video.currentTime;

    reader.drawImage(video, 0, 0, W, H);

    const raw = reader.getImageData(0, 0, W, H).data,
      pixels = [];

    for (let i = 0; i < raw.length; i += 4)
      pixels.push(raw[i], raw[i + 1], raw[i + 2]);

    state = advance(state, pixels);

    drive = state.reduce((a, b) => a + b, 0) / N;

    hostState = neural(hostState, drive, models.circuit);

    if (last?.epoch === epoch && dream) {
      const mae = (p) =>
        p.reduce((sum, v, i) => sum + Math.abs(v - pixels[i]), 0) /
        (255 * p.length);

      predictionError += mae(dream.pixels);
      holdError += mae(last.pixels);
      scored++;
    }

    last = {
      pixels,
      host: hostState.slice(),
      state: state.slice(),
      epoch,
      tick: episodeTick++,
      source: name,
    };

    dream = forecast(last); // Score candidates before adding the current observation.

    memory.push(last);
    window.dispatchEvent(new CustomEvent("studio-classic-frame",{detail:last}));
    if (memory.length > LIMIT) memory.shift();

    tick++;
    draw(dream);
    render();
  }

  async function start(kind, file) {
    window.dispatchEvent(new Event("classic-active"));
    halt();
    const id = token;

    try {
      let source;

      if (kind === "camera") {
        if (!navigator.mediaDevices?.getUserMedia)
          throw Error(
            "Webcam requires a local or HTTPS page and camera permission.",
          );

        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        if (id !== token) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        video.srcObject = stream;
        source = "Webcam";
      } else if (kind === "file") {
        url = URL.createObjectURL(file);
        video.src = url;
        source = file.name;
      } else {
        source = $("senseExample").value;
        video.src = "data:video/webm;base64," + stimuli[source];
      }

      video.loop = kind !== "camera";
      await video.play();
      if (id !== token) return;

      last = null;
      dream = null;
      predictionError = 0;
      holdError = 0;
      scored = 0;
      name = source;
      epoch = crypto.randomUUID();
      episodeTick = 0;
      lastTime = -1;
      state = Array(N).fill(0);
      hostState = [0, 0, 0];

      mode = "Observing " + source;
      status.textContent =
        "Learning observed transitions locally. The right panel forecasts one sensory tick ahead.";

      inputStep();
      interval = setInterval(inputStep, 200);
    } catch (e) {
      if (id === token) {
        halt();
        status.textContent = "Input unavailable. " + e.message;
      }
    }
  }

  $("sensePlay").onclick = () => start("example");

  $("senseCam").onclick = () => start("camera");

  $("senseFile").onchange = (e) => {
    const f = e.target.files[0];
    if (f) start("file", f);
    e.target.value = "";
  };

  $("senseStop").onclick = () => {
    halt();
    status.textContent = "Input stopped. Session memory is available.";
  };

  $("freeDream").onclick = () => {
    window.dispatchEvent(new Event("classic-active"));
    halt();
    const selected = $("recallEpisode").value;
    let context = selected
      ? memory.filter((f) => f.epoch === selected).at(-1)
      : last || memory.at(-1);

    if (!context || !forecast(context)) {
      status.textContent = "Feed several frames from one input first.";
      return;
    }

    mode = "Imagining";
    status.textContent =
      "Autonomous recall. Generated frames feed the next prediction and stay outside observed memory.";

    const step = () => {
      dream = forecast(context);
      context = dream;
      state = dream.state;
      hostState = dream.host;
      draw(dream);
      tick++;
      render();
    };

    step();
    interval = setInterval(step, 200);
  };

  $("exportVisual").onclick = () => {
    const data = JSON.stringify({
      schema: "sonapse-visual-v2",
      width: W,
      height: H,
      frames: memory,
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(
      new Blob([data], { type: "application/json" }),
    );
    link.download = "sonapse-visual-memory.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };
  $("storeMemory").onclick = () => {
    try {
      localStorage.setItem(KEY, JSON.stringify(memory));
      status.textContent =
        "Saved " +
        memory.length +
        " observed frames and neural states in this browser.";
    } catch {
      status.textContent =
        "Browser storage is unavailable or full. Memory remains in this session.";
    }
  };

  $("clearMemory").onclick = () => {
    if (!confirm("Clear this browser’s saved and session visual memory?"))
      return;
    halt();
    memory = [];
    window.dispatchEvent(new Event("studio-classic-clear"));
    last = null;
    dream = null;
    state = Array(N).fill(0);
    hostState = [0, 0, 0];
    tick = 0;
    scored = 0;
    predictionError = 0;
    holdError = 0;
    draw(null);
    render();
    try {
      localStorage.removeItem(KEY);
      status.textContent = "Visual memory cleared.";
    } catch {
      status.textContent =
        "Session cleared. Browser storage could not be accessed.";
    }
  };

  window.addEventListener("studio-active", halt);
  window.addEventListener("pagehide", halt);
  render();
})();
