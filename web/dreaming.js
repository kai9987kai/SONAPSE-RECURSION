// Scene selection is independent of rendering, capture timers and database writes.
class DreamComposer {
  constructor(seed) {
    this.seed=seed>>>0;
    this.random=(()=>{let s=this.seed;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};})();
    this.previous=new Set();
    this.scenes=0;
    this.traveler={form:Math.floor(this.random()*3),orbit:this.random()*Math.PI*2};
  }
  choose(rows, query, novelty, excluded) {
    // Weighted reservoir sampling gives every eligible frame a positive probability.
    let winner, total=0;
    for(const row of rows){
      if(excluded.has(row.id))continue;
      let distance=0;
      if(query)for(let i=0;i<288;i+=3)distance+=Math.abs(row.desc[i]-query[i])/(96*255);
      const affinity=1/(.12+distance);
      const weight=(.08+(1-novelty)*affinity+novelty*(.25+distance))*
        (this.previous.has(row.epoch)? .35:1);
      total+=weight;
      if(this.random()*total<weight)winner=row;
    }
    return winner;
  }
  plan(rows,query,options) {
    const {novelty,theme,startingEpisode}=options;
    const candidates=rows.slice().sort((a,b)=>a.id<b.id?-1:a.id>b.id?1:0);
    let firstPool=candidates;
    if(!this.scenes && startingEpisode){
      const matching=candidates.filter(r=>r.epoch===startingEpisode);
      if(matching.length)firstPool=matching;
    }
    const anchor=this.choose(firstPool,query,novelty,new Set());
    let partners=anchor?candidates.filter(r=>r.epoch!==anchor.epoch):[];
    if(!partners.length)partners=candidates;
    const associate=this.choose(partners,anchor?.desc||query,novelty,new Set([anchor?.id]));
    const fragments=[anchor,associate].filter(Boolean);
    this.previous=new Set(fragments.map(r=>r.epoch));
    const worlds=["Tidal sky","Branching garden","Floating city","Folded passage"];
    const world=theme==="Drift"?worlds[Math.floor(this.random()*worlds.length)]:theme;
    const tone=anchor?Math.atan2(anchor.desc[1]-anchor.desc[2],anchor.desc[0]-anchor.desc[1])*180/Math.PI:360*this.random();
    const hue=(tone+360+novelty*140*this.random())%360;
    if(this.scenes && this.random()<novelty*.65)this.traveler.form=(this.traveler.form+1)%3;
    const focus=fragments.map(r=>{
      let best=-1,cell=0;
      for(let i=0;i<96;i++){
        const k=i*3,neighbor=(i%12 ? i-1 : (i+1)%96)*3;
        const contrast=Math.abs(r.desc[k]-r.desc[neighbor])+Math.abs(r.desc[k+1]-r.desc[neighbor+1])+Math.abs(r.desc[k+2]-r.desc[neighbor+2]);
        if(contrast>best){best=contrast;cell=i;}
      }
      return [(cell%12+.5)/12,(Math.floor(cell/12)+.5)/8];
    });
    return {traveler:{...this.traveler},focus,number:++this.scenes,world,hue,novelty,fragments,eligible:rows.length,
      duration:16+Math.floor(this.random()*10),motifs:Array.from({length:45},()=>Array.from({length:6},()=>this.random()))};
  }
}

function dreamTexture(image) {
  const c=document.createElement("canvas");c.width=image.width;c.height=image.height;
  c.getContext("2d").putImageData(image,0,0);return c;
}

// Render in normalized coordinates so the same scene grammar works through Full HD.
function renderDream(g,plan,textures,age,core,transition) {
  const {world,hue,novelty,motifs}=plan,w=g.canvas.width,h=g.canvas.height;
  const t=age*.065, pulse=(core[0]||0)*.12;
  const color=(offset,sat,light)=>`hsl(${hue+offset} ${sat}% ${light}%)`;
  g.save();g.scale(w/1000,h/600);
  const sky=g.createLinearGradient(0,0,750,600);
  sky.addColorStop(0,color(-32,54,11));sky.addColorStop(.55,color(12,53,40));sky.addColorStop(1,color(75,65,71));
  g.fillStyle=sky;g.fillRect(0,0,1000,600);
  for(const [x,y,size] of motifs){
    g.fillStyle=color(68,64,86);g.beginPath();g.arc(1000*x,360*y,1+2*size,0,Math.PI*2);g.fill();
  }
  const moon=motifs[0];g.fillStyle=color(90,80,81);g.beginPath();
  g.arc(150+moon[0]*700,80+moon[1]*125,24+moon[2]*48,0,2*Math.PI);g.fill();
  if(world==="Tidal sky" || world==="Branching garden"){
    for(let band=0;band<6;band++){
      g.beginPath();g.moveTo(0,610);
      for(let x=0;x<=1010;x+=10){
        const y=260+band*50+Math.sin(x*.007+t+band)*30+Math.sin(x*.016+band+t*.3)*12;
        g.lineTo(x,y);
      }
      g.lineTo(1010,610);g.fillStyle=color(-18+band*6,40,38-band*4);g.fill();
    }
    if(world==="Branching garden"){
      function branch(x,y,len,angle,depth){
        const nx=x+len*Math.cos(angle),ny=y+len*Math.sin(angle);
        g.lineWidth=1+depth*1.2;g.beginPath();g.moveTo(x,y);g.lineTo(nx,ny);g.stroke();
        if(depth){branch(nx,ny,len*.7,angle-.38-pulse,depth-1);branch(nx,ny,len*.68,angle+.48+Math.sin(t)*.09,depth-1);}
        else{g.beginPath();g.arc(nx,ny,3,0,7);g.fill();}
      }
      g.strokeStyle=color(46,46,65);g.fillStyle=color(84,72,78);
      for(const [x,y,size] of motifs.slice(1,7))branch(x*1000,440+y*170,34+size*38,-1.5,5);
    }
  }else if(world==="Floating city"){
    for(const [x,y,a,b,c] of motifs){
      const px=x*1010,base=280+y*340+12*Math.sin(t+a*6),width=14+47*b,height=25+c*130;
      g.fillStyle=color(-20+40*b,45,18+y*13);g.fillRect(px,base-height,width,height);
      g.fillStyle=color(85,70,74);
      for(let wy=base-height+8;wy<base-5;wy+=14)for(let wx=px+5;wx<px+width-3;wx+=10)g.fillRect(wx,wy,3,5);
    }
  }else{
    const vx=460+100*Math.sin(t*.4+moon[0]*6),vy=245+60*pulse;
    for(let i=19;i>=0;i--){const r=Math.exp(-(i+(1-t*.6%1))*.18);
      g.save();g.translate(vx,vy);g.rotate(.25*Math.sin(t*.4+i*.21)*novelty);
      g.strokeStyle=color(i*5,62,35+i*2);g.lineWidth=2+6*r;g.strokeRect(-740*r,-440*r,1480*r,880*r);g.restore();
    }
  }
  // A recurring object retains its orbit across changes of setting and can change form.
  const flight=plan.traveler.orbit+plan.number*.7+t*.4;
  g.save();g.translate(500+330*Math.sin(flight),230+95*Math.cos(flight*1.3)+40*pulse);
  g.rotate(.22*Math.sin(flight));g.fillStyle=color(115,76,80);g.strokeStyle=color(-10,54,24);g.lineWidth=2;
  g.beginPath();
  if(plan.traveler.form===0){g.moveTo(-47,0);g.lineTo(0,-17);g.lineTo(42,0);g.lineTo(0,26);g.closePath();}
  else if(plan.traveler.form===1){g.moveTo(-52,18);g.quadraticCurveTo(-30,-26,0,1);g.quadraticCurveTo(25,-32,55,9);g.quadraticCurveTo(22,0,0,16);g.closePath();}
  else{g.ellipse(0,0,31,24,0,0,7);}
  g.fill();g.stroke();
  g.beginPath();g.moveTo(0,19);g.bezierCurveTo(-8,39,32,52,12+14*Math.sin(t),74);g.stroke();g.restore();
  // Recalled imagery occupies moving regions of the invented scene.
  textures.forEach((texture,j)=>{
    const m=motifs[8+j], x=170+590*m[0]+35*Math.sin(t+m[1]*6),y=150+230*m[2];
    const radius=55+68*m[3]+22*(1-novelty);
    g.save();g.translate(x,y);g.rotate(novelty*.4*Math.sin(t+m[4]*5));g.beginPath();
    g.ellipse(0,0,radius*1.34,radius,0,0,Math.PI*2);g.clip();
    const crop=.8-.38*novelty;
    const sx=texture.width*(1-crop)*plan.focus[j][0],sy=texture.height*(1-crop)*plan.focus[j][1];
    g.globalAlpha=.96-novelty*.24;
    g.drawImage(texture,sx,sy,texture.width*crop,texture.height*crop,-radius*1.34,-radius,2.68*radius,2*radius);g.restore();
  });
  g.restore();
  if(transition && age<5){g.save();g.globalAlpha=(5-age)/6;g.drawImage(transition,0,0,w,h);g.restore();}
}
