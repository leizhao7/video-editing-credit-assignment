const R=__ROWS__, NV=new Set(__NOVEL__);
const NVT=new Set(__NOVELTR__);
const g=document.getElementById('g');
let f='all', playing=null;
function ok(r){
 if(f==='all')return true;
 if(f==='J'||f==='L')return r.sp===f;
 if(f==='dip')return r.tr!=='cut';
 if(f==='pc')return !!r.pc;
 if(f==='nv')return NV.has(r.tc);
 if(f==='nvt')return NVT.has(r.tr);
 if(f==='plain')return r.tc==='plain';
 return r.mu.includes(f);
}
function stills(id){
 return '<div class="st"><img src="cutframes/c'+id+'a.jpg" loading="lazy">'+
        '<img src="cutframes/c'+id+'b.jpg" loading="lazy"></div>'+
        '<span class="lb l">切点前</span><span class="lb r">切点后</span>'+
        '<span class="play">▶ 播放 · 前后 1.8s · 带声</span>';
}
function toggle(pair,id){
 if(pair.querySelector('video')){ pair.innerHTML=stills(id); playing=null; return; }
 if(playing) playing.el.innerHTML=stills(playing.id);
 pair.innerHTML='<video src="cutclips/c'+id+'.mp4" autoplay controls loop playsinline></video><span class="mid"></span>';
 playing={el:pair,id:id};
}
function render(){
 g.innerHTML='';
 R.filter(ok).forEach(r=>{
  const id=String(r.i).padStart(3,'0');
  const p=Object.entries(r.trp).map(([k,v])=>k+' '+v+'s').join(' ');
  const el=document.createElement('div'); el.className='c';
  el.innerHTML='<div class="hd"><span class="n">#'+r.i+'</span>'+
   '<span class="tt">'+r.t+'s &nbsp; '+r.A+' → '+r.B+'</span></div>'+
   '<div class="pair">'+stills(id)+'</div>'+
   '<div class="tags">'+
    '<span class="t tr">'+r.tr+(p?' '+p:'')+'</span>'+
    '<span class="t tc'+(NV.has(r.tc)?' nv':'')+'">'+r.tc+(NV.has(r.tc)?' ★':'')+'</span>'+
    '<span class="t sp'+(r.sp==='hard'?' h':'')+'">语音 '+r.sp+(r.off?' ±'+r.off+'s':'')+'</span>'+
    r.mu.map(m=>'<span class="t mu">音乐 '+m+'</span>').join('')+
    (r.pc===true?'<span class="t pc">画面连续'+(r.pcf?'?':'')+'</span>':'')+
   '</div>'+
   (r.line?'<div class="line">「'+r.line+'」</div>':'')+
   (r.note?'<div class="note">'+r.note+'</div>':'');
  const pair=el.querySelector('.pair');
  pair.onclick=e=>{ if(e.target.tagName!=='VIDEO') toggle(pair,id); };
  g.appendChild(el);
 });
 playing=null;
}
document.querySelectorAll('.f button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.f button').forEach(x=>x.classList.remove('on'));
 b.classList.add('on'); f=b.dataset.f; render(); scrollTo(0,0);});
render();