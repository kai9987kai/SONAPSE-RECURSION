// Full-archive visual recall. Large images stay on disk until selected for synthesis.
(() => {
  const $ = (id) => document.getElementById(id);
  const storageName = "sonapse-visual-studio-v1";
  const inputs = new Map(),
    records = new Map(),
    legacy = new Map();
  const labels = {
    orbit: "Orbit",
    ripple: "Ripple",
    crossing: "Crossing",
    julia: "Julia set",
    mandelbrot: "Mandelbrot zoom",
    carpet: "Sierpiński carpet",
    interference: "Wave interference",
    ribbons: "Flowing ribbons",
  };
  for (const [name, label] of Object.entries(labels))
    $("mixExample").add(new Option(label, name));
  const scene = $("mixScene"),
    output = $("mixDream"),
    net = $("mixNetwork");
  const ctx = scene.getContext("2d"),
    paint = output.getContext("2d", { willReadFrequently: true });
  const tiny = document.createElement("canvas");
  tiny.width = 12;
  tiny.height = 8;
  const small = tiny.getContext("2d", { willReadFrequently: true });
  const thumb = document.createElement("canvas");
  thumb.width = 20;
  thumb.height = 12;
  const thumbCtx = thumb.getContext("2d");
  let db,
    timer,
    busy = false,
    generation = 0,
    lifecycle = 0,
    episode = "",
    step = 0,
    state = new Float32Array(96),
    host = [0, 0, 0];
  let mode = "Idle",
    lastDream = null,
    bytes = 0,
    selectedCount = 0;
  const flash = (text) => {
    $("studioStatus").textContent = text;
  };
  const decoded = new Map();
  let firstForecast = null,
    previousScene = null,
    modelError = 0,
    holdError = 0,
    comparisons = 0;
  const bus =
    typeof BroadcastChannel === "function"
      ? new BroadcastChannel(storageName)
      : null;
  function query(req) {
    return new Promise((resolve, reject) => {
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  function settled(tx) {
    return new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onabort = tx.onerror = () =>
        reject(tx.error || Error("Storage transaction failed"));
    });
  }
  function status() {
    $("studioArchive").textContent =
      mode +
      " · " +
      inputs.size +
      " active inputs · " +
      records.size +
      " studio + " +
      legacy.size +
      " classic observations · " +
      selectedCount +
      " memories used in last forecast · " +
      (bytes / 1048576).toFixed(1) +
      " MiB of studio images" +
      (comparisons
        ? " · Next-frame / hold MAE " +
          (modelError / comparisons).toFixed(3) +
          " / " +
          (holdError / comparisons).toFixed(3)
        : "");
  }
  function listEpisodes() {
    const sel = $("mixEpisode"),
      selected = sel.value,
      groups = new Map();
    for (const r of [...records.values(), ...legacy.values()])
      groups.set(r.epoch, r.source);
    sel.replaceChildren(
      new Option("Latest", ""),
      ...Array.from(
        groups,
        ([id, label], i) => new Option(i + 1 + ". " + label, id),
      ),
    );
    if (groups.has(selected)) sel.value = selected;
  }
  async function refresh() {
    const rows = await query(
      db.transaction("index").objectStore("index").getAll(),
    );
    records.clear();
    bytes = 0;
    for (const r of rows) {
      records.set(r.id, r);
      bytes += r.bytes;
    }
    listEpisodes();
    status();
  }
  const ready = (async () => {
    const req = indexedDB.open(storageName, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore("index", { keyPath: "id" });
      req.result.createObjectStore("images");
    };
    db = await query(req);
    await refresh();
    return true;
  })().catch((e) => {
    flash("Studio archive unavailable: " + e.message);
    return false;
  });
  if (bus)
    bus.onmessage = async (e) => {
      if (!(await ready)) return;
      if (e.data?.id) {
        const r = await query(
          db.transaction("index").objectStore("index").get(e.data.id),
        );
        if (r && !records.has(r.id)) {
          records.set(r.id, r);
          bytes += r.bytes;
          if (!r.tick) listEpisodes();
          status();
        }
      } else {
        decoded.clear();
        await refresh();
      }
    };
  function describe(canvas) {
    small.drawImage(canvas, 0, 0, 12, 8);
    const p = small.getImageData(0, 0, 12, 8).data,
      desc = new Uint8Array(288);
    for (let i = 0; i < 96; i++) desc.set(p.subarray(i * 4, i * 4 + 3), i * 3);
    return desc;
  }
  function legacyCanvas(r) {
    const img = thumbCtx.createImageData(20, 12);
    for (let i = 0; i < 240; i++) {
      img.data.set(r.pixels.slice(i * 3, i * 3 + 3), i * 4);
      img.data[i * 4 + 3] = 255;
    }
    thumbCtx.putImageData(img, 0, 0);
    return thumb;
  }
  function addClassic(f) {
    const id = "classic:" + f.epoch + ":" + f.tick;
    if (!legacy.has(id) && f.pixels?.length === 720 && f.host?.length === 3) {
      const r = { ...f, id, desc: describe(legacyCanvas(f)), time: 0 };
      legacy.set(id, r);
    }
  }
  function savedClassic() {
    try {
      for (const f of JSON.parse(
        localStorage.getItem("sonapse.visual.episodes.v2") || "[]",
      ))
        addClassic(f);
    } catch {}
  }
  savedClassic();
  window.addEventListener("studio-classic-frame", (e) => {
    addClassic(e.detail);
    status();
  });
  window.addEventListener("studio-classic-clear", () => {
    legacy.clear();
    listEpisodes();
    status();
  });
  window.addEventListener("storage", (e) => {
    if (e.key === "sonapse.visual.episodes.v2") {
      if (e.newValue) savedClassic();
      else legacy.clear();
      listEpisodes();
      status();
    }
  });
  function advance(desc, prev, core) {
    const next = Float32Array.from(prev, (v, i) => {
      const k = i * 3,
        light =
          (desc[k] * 0.213 + desc[k + 1] * 0.715 + desc[k + 2] * 0.072) / 255;
      return Math.tanh(
        0.4 * v +
          1.4 * (light - 0.5) +
          0.48 * prev[(i + 95) % 96] -
          0.29 * prev[(i + 84) % 96] +
          0.21 * prev[(i + 59) % 96],
      );
    });
    return [
      next,
      neural(core, next.reduce((a, b) => a + b, 0) / 96, models.circuit),
    ];
  }
  function updateState(desc) {
    [state, host] = advance(desc, state, host);
    drawNetwork();
  }
  function drawNetwork() {
    const g = net.getContext("2d");
    g.clearRect(0, 0, 480, 288);
    state.forEach((v, i) => {
      const x = 20 + (i % 12) * 40,
        y = 18 + Math.floor(i / 12) * 36;
      for (const [delta, w] of [
        [1, 0.48],
        [12, -0.29],
        [37, 0.21],
      ]) {
        const j = (i + delta) % 96;
        g.strokeStyle = w > 0 ? "#8de6c1" : "#f8ad84";
        g.globalAlpha = 0.08 + 0.55 * Math.abs(v * w);
        g.beginPath();
        g.moveTo(x, y);
        g.lineTo(20 + (j % 12) * 40, 18 + Math.floor(j / 12) * 36);
        g.stroke();
      }
      g.globalAlpha = 1;
      g.fillStyle = v >= 0 ? "#8de6c1" : "#f8ad84";
      g.beginPath();
      g.arc(x, y, 2 + 6 * Math.abs(v), 0, 7);
      g.fill();
    });
  }
  function resetEpisode() {
    generation++;
    episode = crypto.randomUUID();
    step = 0;
    state = new Float32Array(96);
    host = [0, 0, 0];
    lastDream = null;
    firstForecast = null;
    previousScene = null;
    comparisons = 0;
    modelError = 0;
    holdError = 0;
  }
  function release(item) {
    item.video.pause();
    item.video.srcObject?.getTracks().forEach((t) => t.stop());
    item.video.srcObject = null;
    item.video.removeAttribute("src");
    item.video.load();
    if (item.url) URL.revokeObjectURL(item.url);
  }
  function showInputs() {
    $("studioSources").replaceChildren(
      ...Array.from(inputs, ([id, item]) => {
        const button = document.createElement("button");
        button.textContent = "Remove " + item.name;
        button.onclick = () => {
          release(item);
          inputs.delete(id);
          resetEpisode();
          showInputs();
          if (!inputs.size) stop();
        };
        return button;
      }),
    );
    status();
  }
  let sleep=null;
  function dreamControls(){
    const active=mode==="Dreaming";
    $("dreamWake").disabled=!active;
    $("dreamPause").disabled=!active;
    $("dreamPause").textContent=active && sleep.paused?"Resume dream":"Pause dream";
    $("dreamStep").disabled=!active || !sleep?.paused;
    $("dreamJournal").disabled=!sleep?.journal.length;
    $("dreamStrip").disabled=!sleep?.stills.length;
  }
  async function dreamFrame(id,singleStep){
    const session=sleep;
    if(!session.plan || session.age>=session.plan.duration){
      const rows=[...legacy.values(),...records.values()];
      const plan=session.composer.plan(rows,session.cue,{
        novelty:Number($("dreamFreedom").value)/100,
        theme:$("dreamTheme").value,startingEpisode:session.startingEpisode
      });
      const textures=await Promise.all(plan.fragments.map(async r=>dreamTexture(await pixels(r))));
      if(id!==generation || session!==sleep)return;
      session.transition=session.frame?dreamTexture(paint.getImageData(0,0,output.width,output.height)):null;
      session.plan=plan;session.textures=textures;session.age=0;
      selectedCount=plan.eligible;
      session.journal.push({scene:plan.number,firstFrame:session.frame+1,world:plan.world,
        travelerForm:["kite","bird","lantern"][plan.traveler.form],freedom:plan.novelty,eligibleMemories:plan.eligible,
        fragments:plan.fragments.map(r=>({id:r.id,source:r.source,epoch:r.epoch,tick:r.tick}))});
      if(session.journal.length>96)session.journal.shift();
    }
    if(session.paused && !singleStep)return;
    renderDream(paint,session.plan,session.textures,session.age,host,session.transition);
    const cue=describe(output);
    updateState(cue);session.cue=cue;session.age++;session.frame++;
    if(session.age>=5)session.transition=null;
    if(session.age===1 || session.age===6){
      const still=document.createElement("canvas");still.width=200;still.height=Math.round(200*output.height/output.width);
      still.getContext("2d").drawImage(output,0,0,still.width,still.height);
      const shot={still,world:session.plan.world,scene:session.plan.number};
      if(session.age===6)session.stills.pop();
      session.stills.push(shot);if(session.stills.length>6)session.stills.shift();
      const strip=$("dreamGallery");strip.replaceChildren(...session.stills.map(item=>{
        const fig=document.createElement("figure"),label=document.createElement("figcaption");
        label.textContent=item.scene+" · "+item.world;fig.append(item.still,label);return fig;
      }));
    }
    $("mixImage").disabled=false;
    const plan=session.plan;
    $("dreamReadout").textContent="Seed "+session.composer.seed+" · Scene "+plan.number+" · Frame "+session.frame+" · "+plan.world+" · "+
      (plan.fragments.length?"Recalled "+plan.fragments.map(r=>r.source).join(" + "):"Entirely procedural scene");
    dreamControls();
  }
  $("dreamStart").onclick=async()=>{
    const raw=$("dreamSeed").value;
    if(!$("dreamSeed").checkValidity()){flash("Choose a whole-number seed between 0 and 4294967295.");return;}
    const seed=raw===""?crypto.getRandomValues(new Uint32Array(1))[0]:Number(raw);
    stop();activate();const id=generation;
    await ready;
    if(id!==generation)return;
    sleep={composer:new DreamComposer(seed),startingEpisode:$("mixEpisode").value,paused:false,age:0,frame:0,journal:[],stills:[],cue:null,plan:null};
    mode="Dreaming";$("dreamGallery").replaceChildren();dreamControls();
    $("dreamReadout").textContent="Entering dream…";
    flash("Dream frames feed the visual connectome. Setting and freedom changes apply to the next scene.");
    startLoop();
  };
  $("dreamPause").onclick=()=>{
    if(mode!=="Dreaming")return;
    sleep.paused=!sleep.paused;dreamControls();
    if(!sleep.paused)void tick();
  };
  $("dreamStep").onclick=()=>{if(mode==="Dreaming" && sleep.paused)void tick(true);};
  $("dreamWake").onclick=()=>{stop();flash("Awake. The final image and dream journal are available.");};
  $("dreamStrip").onclick=()=>{
    if(!sleep?.stills.length)return;
    const strip=document.createElement("canvas");strip.width=200*sleep.stills.length;
    const height=Math.max(...sleep.stills.map(s=>s.still.height));strip.height=height+32;
    const g=strip.getContext("2d");g.fillStyle="#142820";g.fillRect(0,0,strip.width,strip.height);
    const filename="dream-strip-"+sleep.composer.seed+".png";
    sleep.stills.forEach(({still,scene,world},i)=>{
      g.drawImage(still,i*200,0);g.fillStyle="#e6f2e4";g.font="14px sans-serif";g.fillText(scene+" · "+world,8+i*200,height+21);
    });
    strip.toBlob(blob=>{if(!blob)return;const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);},"image/png");
  };
  $("dreamJournal").onclick=()=>{
    if(!sleep)return;
    const journal={schema:"sonapse-dream-v1",seed:sleep.composer.seed,startingEpisode:sleep.startingEpisode,
      frames:sleep.frame,scenes:sleep.composer.scenes,synthetic:true,entries:sleep.journal};
    const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(journal,null,2)],{type:"application/json"}));
    a.download="dream-"+journal.seed+".json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  };
  function stop() {
    lifecycle++;
    resetEpisode();
    clearInterval(timer);
    for (const item of inputs.values()) release(item);
    inputs.clear();
    if(mode==="Dreaming" && sleep){sleep.textures=[];sleep.transition=null;sleep.plan=null;}
    mode = "Idle";dreamControls();
    showInputs();
  }
  function activate() {
    window.dispatchEvent(new Event("studio-active"));
  }
  window.addEventListener("classic-active", stop);
  window.addEventListener("pagehide", () => {
    stop();
    bus?.close();
  });
  function startLoop() {
    clearInterval(timer);
    timer = setInterval(() => void tick(), 240);
    void tick();
  }
  async function add(kind, value) {
    const ticket = lifecycle;
    if (!(await ready) || ticket !== lifecycle) return;
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.loop = kind !== "camera";
    const item = {
      video,
      name:
        kind === "camera"
          ? "Webcam"
          : kind === "file"
            ? value.name
            : labels[value],
      url: null,
      time: -1,
    };
    try {
      if (kind === "camera") {
        if (!navigator.mediaDevices?.getUserMedia)
          throw Error("Camera access is unavailable");
        video.srcObject = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } else if (kind === "stream") {
        const remote = new URL(value);
        if (!["http:", "https:"].includes(remote.protocol))
          throw Error("Choose an HTTP or HTTPS media URL.");
        item.name = "Stream · " + remote.hostname;
        video.crossOrigin = "anonymous";
        video.src = remote.href;
      } else if (kind === "file") {
        item.url = URL.createObjectURL(value);
        video.src = item.url;
      } else video.src = "data:video/webm;base64," + STUDIO_CLIPS[value];
      if (ticket !== lifecycle) {
        release(item);
        return;
      }
      await video.play();
      if (ticket !== lifecycle) {
        release(item);
        return;
      }
      if (kind === "stream") {
        const test = document.createElement("canvas");
        test.width = test.height = 1;
        const probe = test.getContext("2d");
        probe.drawImage(video, 0, 0, 1, 1);
        probe.getImageData(0, 0, 1, 1);
      }
      activate();
      inputs.set(crypto.randomUUID(), item);
      mode = "Observing";
      resetEpisode();
      showInputs();
      flash(
        "Saving composite observations locally. Remove an input to change the scene.",
      );
      startLoop();
    } catch (e) {
      release(item);
      flash("Could not add " + item.name + ". " + e.message);
    }
  }
  function resize() {
    decoded.clear();
    $("mixImage").disabled = true;
    const w = Number($("mixResolution").value);
    for (const c of [scene, output]) {
      c.width = w;
      c.height = w >= 1280 ? (w * 9) / 16 : (w * 3) / 5;
      c.style.aspectRatio = c.width + "/" + c.height;
    }
    resetEpisode();
  }
  function compose() {
    const list = [...inputs.values()];
    if (
      !list.some(
        (s) => s.video.readyState >= 2 && s.time !== s.video.currentTime,
      )
    )
      return false;
    if (list.some((s) => s.time > s.video.currentTime)) resetEpisode();
    ctx.fillStyle = "#122824";
    ctx.fillRect(0, 0, scene.width, scene.height);
    const cols = Math.ceil(Math.sqrt(list.length)),
      rows = Math.ceil(list.length / cols);
    list.forEach((s, i) => {
      if (s.video.readyState < 2) return;
      s.time = s.video.currentTime;
      const w = scene.width / cols,
        h = scene.height / rows,
        scale = Math.min(w / s.video.videoWidth, h / s.video.videoHeight);
      const dw = s.video.videoWidth * scale,
        dh = s.video.videoHeight * scale;
      if (list.length > 1 && $("mixSeparate").checked) {
        s.capture ??= document.createElement("canvas");
        if (
          s.capture.width !== scene.width ||
          s.capture.height !== scene.height
        ) {
          s.capture.width = scene.width;
          s.capture.height = scene.height;
        }
        s.capture
          .getContext("2d")
          .drawImage(s.video, 0, 0, scene.width, scene.height);
      }
      ctx.drawImage(
        s.video,
        (i % cols) * w + (w - dw) / 2,
        Math.floor(i / cols) * h + (h - dh) / 2,
        dw,
        dh,
      );
    });
    return true;
  }
  async function remember(desc, id) {
    const captured = {
      id: episode + ":" + step,
      epoch: episode,
      tick: step,
      source: [...inputs.values()].map((s) => s.name).join(" + "),
      desc,
      host: [...host],
      state: state.slice(),
      time: Date.now(),
      width: scene.width,
      height: scene.height,
    };
    const batch = [{ row: captured, canvas: scene }];
    if (inputs.size > 1 && $("mixSeparate").checked) {
      for (const [key, item] of inputs) {
        if (!item.capture || item.video.readyState < 2) continue;
        if (item.snapshotEpoch !== episode) {
          item.snapshotEpoch = episode;
          item.snapshotTick = 0;
          item.nodes = new Float32Array(96);
          item.core = [0, 0, 0];
        }
        const features = describe(item.capture);
        [item.nodes, item.core] = advance(features, item.nodes, item.core);
        const sourceEpoch = episode + "-" + key,
          sourceTick = item.snapshotTick++;
        batch.push({
          row: {
            ...captured,
            epoch: sourceEpoch,
            tick: sourceTick,
            id: sourceEpoch + ":" + sourceTick,
            source: item.name,
            desc: features,
            host: item.core,
            state: item.nodes,
          },
          canvas: item.capture,
        });
      }
    }
    const blobs = await Promise.all(
      batch.map(
        ({ canvas }) =>
          new Promise((resolve) => canvas.toBlob(resolve, "image/webp", 0.86)),
      ),
    );
    if (blobs.some((b) => !b)) throw Error("Frame encoding failed");
    if (id !== generation) return;
    const tx = db.transaction(["index", "images"], "readwrite"),
      done = settled(tx);
    batch.forEach(({ row }, i) => {
      row.bytes = blobs[i].size;
      tx.objectStore("index").add(row);
      tx.objectStore("images").add(blobs[i], row.id);
    });
    await done;
    for (const { row } of batch) {
      records.set(row.id, row);
      bytes += row.bytes;
      bus?.postMessage({ id: row.id });
    }
    if (id === generation) step++;
    if (captured.tick === 0) listEpisodes();
  }
  async function pixels(r) {
    const key = r.id + "@" + output.width;
    if (decoded.has(key)) return decoded.get(key);
    const c = document.createElement("canvas");
    c.width = output.width;
    c.height = output.height;
    const g = c.getContext("2d", { willReadFrequently: true });
    if (r.pixels) g.drawImage(legacyCanvas(r), 0, 0, c.width, c.height);
    else {
      const blob = await query(
        db.transaction("images").objectStore("images").get(r.id),
      );
      if (!blob) throw Error("An archived image was removed");
      const image = await createImageBitmap(blob);
      g.drawImage(image, 0, 0, c.width, c.height);
      image.close();
    }
    const imageData = g.getImageData(0, 0, c.width, c.height);
    decoded.set(key, imageData);
    const cacheLimit = Math.max(
      1,
      Math.min(12, Math.floor((48 * 1024 * 1024) / (c.width * c.height * 4))),
    );
    while (decoded.size > cacheLimit)
      decoded.delete(decoded.keys().next().value);
    return imageData;
  }
  async function predict(context, desc, id, core) {
    const all = [...legacy.values(), ...records.values()];
    selectedCount = all.length;
    if (!all.length) return;
    const average = new Float64Array(288);
    let weightSum = 0;
    const top = [];
    for (const r of all) {
      let error = 0;
      for (let k = 0; k < 288; k++)
        error += Math.abs(desc[k] - r.desc[k]) / (255 * 288);
      for (let k = 0; k < 3; k++) error += 0.04 * Math.abs(core[k] - r.host[k]);
      const weight = 1 / (0.035 + error) ** 2;
      weightSum += weight;
      for (let k = 0; k < 288; k++) average[k] += weight * r.desc[k];
      const successor = (r.pixels ? legacy : records).get(
        (r.pixels ? "classic:" : "") + r.epoch + ":" + (r.tick + 1),
      );
      if (
        successor &&
        (top.length < 3 || weight > top[top.length - 1].weight)
      ) {
        top.push({ r, successor, weight });
        top.sort((a, b) => b.weight - a.weight);
        if (top.length > 3) top.pop();
      }
    }
    const motions = await Promise.all(
      top.map(async (m) => ({
        ...m,
        a: await pixels(m.r),
        b: await pixels(m.successor),
      })),
    );
    if (id !== generation) return;
    const w = output.width,
      h = output.height,
      result = paint.createImageData(w, h);
    const sum = motions.reduce((s, m) => s + m.weight, 0);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4,
          k = (Math.floor((y * 8) / h) * 12 + Math.floor((x * 12) / w)) * 3;
        for (let c = 0; c < 3; c++) {
          const v = context.data[i + c];
          let forecast = v;
          for (const m of motions)
            forecast +=
              (m.weight / sum) *
              (0.8 * (m.b.data[i + c] - m.a.data[i + c]) +
                0.2 * (m.b.data[i + c] - v));
          result.data[i + c] =
            0.96 * forecast + (0.04 * average[k + c]) / weightSum;
        }
        result.data[i + 3] = 255;
      }
    return result;
  }
  async function rollout(context, desc, id, autonomous) {
    let nodes = state,
      core = host;
    const hops = Number($("mixHorizon").value);
    for (let i = 0; i < hops; i++) {
      const generated = await predict(context, desc, id, core);
      if (!generated || id !== generation) return;
      if (!i && !autonomous) firstForecast = generated;
      paint.putImageData(generated, 0, 0);
      context = generated;
      desc = describe(output);
      [nodes, core] = advance(desc, nodes, core);
    }
    lastDream = context;
    $("mixImage").disabled = false;
    if (autonomous) {
      state = nodes;
      host = core;
      drawNetwork();
    }
  }
  function mae(a, b) {
    let error = 0;
    for (let i = 0; i < a.data.length; i += 4)
      for (let c = 0; c < 3; c++)
        error += Math.abs(a.data[i + c] - b.data[i + c]);
    return error / (a.width * a.height * 3 * 255);
  }
  async function tick(singleStep=false) {
    if (busy || mode === "Idle") return;
    busy = true;
    const runGeneration=generation;
    try {
      if(mode==="Dreaming"){
        if(!sleep.paused || singleStep)await dreamFrame(generation,singleStep);
      }else if (mode === "Observing") {
        if (!compose()) return;
        const id = generation,
          d = describe(scene);
        updateState(d);
        const actual = ctx.getImageData(0, 0, scene.width, scene.height);
        if (firstForecast && previousScene) {
          modelError += mae(firstForecast, actual);
          holdError += mae(previousScene, actual);
          comparisons++;
        }
        previousScene = actual;
        await rollout(actual, d, id, false);
        if (id === generation) await remember(d, id);
      } else if (lastDream) {
        await rollout(lastDream, describe(output), generation, true);
      }
      status();
    } catch (e) {
      if(runGeneration!==generation)return;
      stop();
      flash(
        "Studio paused. " +
          e.message +
          " Previously committed observations remain in the archive.",
      );
    } finally {
      busy = false;
    }
  }
  $("mixSeparate").onchange = resetEpisode;
  $("mixAdd").onclick = () => add("example", $("mixExample").value);
  $("mixStream").onclick = () => add("stream", $("mixUrl").value);
  $("mixCamera").onclick = () => add("camera");
  $("mixFiles").onchange = (e) => {
    for (const f of e.target.files) add("file", f);
    e.target.value = "";
  };
  $("mixStop").onclick = () => {
    stop();
    flash("Inputs released. Memory is available for imagination.");
  };
  $("mixResolution").onchange = () => {
    const dreaming = mode === "Imagining";
    resize();
    if (dreaming) $("mixImagine").click();
  };
  $("mixImagine").onclick = async () => {
    stop();
    activate();
    if (!(await ready)) return;
    listEpisodes();
    const chosen = $("mixEpisode").value;
    const candidates = [...legacy.values(), ...records.values()].filter(
      (r) => !chosen || r.epoch === chosen,
    );
    const seed = candidates
      .sort((a, b) => a.time - b.time || a.tick - b.tick)
      .at(-1);
    if (!seed) {
      flash("Capture an observation first.");
      return;
    }
    const id = generation;
    try {
      const image = await pixels(seed);
      if (id !== generation) return;
      lastDream = image;
      paint.putImageData(image, 0, 0);
      host = Array.from(seed.host);
      if (seed.state?.length === 96) state = Float32Array.from(seed.state);
      mode = "Imagining";
      flash(
        "Autonomous rollout from " +
          seed.source +
          ". Generated frames stay outside the observation archive.",
      );
      startLoop();
    } catch (e) {
      flash(e.message);
    }
  };
  $("mixClear").onclick = async () => {
    if (
      !(await ready) ||
      !confirm(
        "Erase the studio archive on this browser? Classic memories have a separate clear control.",
      )
    )
      return;
    stop();
    while (busy) await new Promise((r) => setTimeout(r, 10));
    try {
      const tx = db.transaction(["index", "images"], "readwrite"),
        done = settled(tx);
      for (const s of ["index", "images"]) tx.objectStore(s).clear();
      await done;
      selectedCount = 0;sleep=null;dreamControls();$("dreamGallery").replaceChildren();$("dreamReadout").textContent="";
      ctx.clearRect(0, 0, scene.width, scene.height);
      paint.clearRect(0, 0, output.width, output.height);
      drawNetwork();
      $("mixImage").disabled = true;
      decoded.clear();
      await refresh();
      bus?.postMessage("changed");
      flash("Studio archive cleared.");
    } catch (e) {
      flash(e.message);
    }
  };
  $("mixImage").onclick = () =>
    output.toBlob((blob) => {
      if (!blob) return;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = (mode==="Dreaming"?"dream-":"imagined-") + output.width + "x" + output.height + ".png";
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }, "image/png");
  $("mixExport").onclick = async () => {
    if (!(await ready)) return;
    try {
      const rows = [];
      for (const r of [...records.values()]) {
        const blob = await query(
          db.transaction("images").objectStore("images").get(r.id),
        );
        const data = await new Promise((ok, fail) => {
          const reader = new FileReader();
          reader.onload = () => ok(reader.result);
          reader.onerror = () => fail(reader.error);
          reader.readAsDataURL(blob);
        });
        rows.push({
          ...r,
          state: [...r.state],
          desc: [...r.desc],
          image: data,
        });
      }
      const a = document.createElement("a");
      a.download = "studio-memory.json";
      a.href = URL.createObjectURL(
        new Blob(
          [JSON.stringify({ schema: "sonapse-studio-v1", frames: rows })],
          { type: "application/json" },
        ),
      );
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (e) {
      flash("Export failed. " + e.message);
    }
  };
  resize();
  listEpisodes();
  status();
})();
