// Simple tag autocomplete for the positive prompt textarea
// Loads tags from all csv files in a1111-sd-webui-tagcomplete/tags and shows suggestions while typing

(function(){
    // caret position helper from textarea-caret-position
    function getCaretCoordinates(element, position){
        const properties=["direction","boxSizing","width","height","overflowX","overflowY","borderTopWidth","borderRightWidth","borderBottomWidth","borderLeftWidth","borderStyle","paddingTop","paddingRight","paddingBottom","paddingLeft","fontStyle","fontVariant","fontWeight","fontStretch","fontSize","fontSizeAdjust","lineHeight","fontFamily","textAlign","textTransform","textIndent","textDecoration","letterSpacing","wordSpacing","tabSize","MozTabSize"];
        const div=document.createElement('div');
        document.body.appendChild(div);
        const style=div.style;
        const computed=window.getComputedStyle?window.getComputedStyle(element):element.currentStyle;
        const isInput=element.nodeName==='INPUT';
        style.whiteSpace='pre-wrap';
        if(!isInput) style.wordWrap='break-word';
        style.position='absolute';
        style.visibility='hidden';
        properties.forEach(prop=>{
            if(isInput && prop==='lineHeight'){
                if(computed.boxSizing==='border-box'){
                    const height=parseInt(computed.height);
                    const outerHeight=parseInt(computed.paddingTop)+parseInt(computed.paddingBottom)+parseInt(computed.borderTopWidth)+parseInt(computed.borderBottomWidth);
                    const targetHeight=outerHeight+parseInt(computed.lineHeight);
                    if(height>targetHeight) style.lineHeight=height-outerHeight+'px';
                    else if(height===targetHeight) style.lineHeight=computed.lineHeight;
                    else style.lineHeight=0;
                } else {
                    style.lineHeight=computed.height;
                }
            } else {
                style[prop]=computed[prop];
            }
        });
        style.overflow='hidden';
        div.textContent=element.value.substring(0,position);
        if(isInput) div.textContent=div.textContent.replace(/\s/g,'\u00a0');
        const span=document.createElement('span');
        span.textContent=element.value.substring(position)||'.';
        div.appendChild(span);
        const coordinates={
            top:span.offsetTop+parseInt(computed.borderTopWidth),
            left:span.offsetLeft+parseInt(computed.borderLeftWidth),
            height:parseInt(computed.lineHeight)
        };
        document.body.removeChild(div);
        return coordinates;
    }
    const TAG_FILES = (window.tag_csv_files && Array.isArray(window.tag_csv_files)) ? window.tag_csv_files : [
        'a1111-sd-webui-tagcomplete/tags/danbooru.csv',
        'a1111-sd-webui-tagcomplete/tags/extra-quality-tags.csv'
    ];
    const MAX_RESULTS = 5;
    let tags = [];
    let container; // suggestion container
    let selected = -1;
    let skipInput = false;

    function parseCSV(line){
        const result=[];
        let cur='';
        let q=false;
        for(let i=0;i<line.length;i++){
            const ch=line[i];
            if(ch==='"'){ q=!q; continue; }
            if(ch===',' && !q){ result.push(cur); cur=''; }
            else cur+=ch;
        }
        result.push(cur);
        return result;
    }

    function formatCount(num){
        if(!num) return '';
        return Intl.NumberFormat('en', {notation:'compact', maximumFractionDigits:1}).format(num);
    }

    async function loadTags(){
        const loaded = [];
        try {
            for(const file of TAG_FILES){
                const resp = await fetch('file=' + file);
                if(!resp.ok) continue;
                const text = await resp.text();
                text.split(/\n/).forEach(line=>{
                    line=line.trim();
                    if(!line) return;
                    const p=parseCSV(line);
                    if(p.length>=1){
                        const entry = {tag:p[0]};
                        if(p.length>=3){
                            const c=parseInt(p[2]);
                            if(!isNaN(c)) entry.count = c;
                            else entry.meta = p[2];
                        }
                        if(!entry.meta && p.length>=2){
                            const m = parseInt(p[1]);
                            if(isNaN(m)) entry.meta = p[1];
                        }
                        loaded.push(entry);
                    }
                });
            }
            const seen=new Set();
            loaded.forEach(t=>{ if(!seen.has(t.tag)){ tags.push(t); seen.add(t.tag);} });
        } catch(err){
            console.error('Failed to load tags', err);
        }
    }

    function createContainer(area){
        container = document.createElement('div');
        container.style.position = 'absolute';
        container.style.background = '#1e1e1e';
        container.style.color = '#fff';
        container.style.border = '1px solid #444';
        container.style.zIndex = 10000;
        container.style.display = 'none';
        container.style.maxHeight = 'none';
        container.style.overflowY = 'hidden';
        container.style.borderRadius = '8px';
        container.style.boxShadow = '0 2px 5px rgba(0,0,0,0.6)';
        container.style.minWidth = '150px';
        document.body.appendChild(container);
    }

    function showSuggestions(area){
        const cursorPos = area.selectionStart;
        const text = area.value.substring(0, cursorPos);
        const fragment = text.split(/[,\n]/).pop().trim();
        if(fragment.length === 0){
            container.style.display = 'none';
            return;
        }
        const lower = fragment.toLowerCase();
        const results = tags.filter(t => t.tag.startsWith(lower));
        if(results.length === 0){
            container.style.display = 'none';
            return;
        }
        results.sort((a,b)=>b.count - a.count);
        const limited = results.slice(0, MAX_RESULTS);
        container.innerHTML = '';
        const colors = ['#ffa500','#f48fb1','#f06292','#ec407a','#e91e63'];
        limited.forEach((t,i)=>{
            const div = document.createElement('div');
            const pre = t.tag.substring(0, fragment.length);
            const post = t.tag.substring(fragment.length);
            const color = colors[(i+1)%colors.length];
            div.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center"><span><span style="color:${colors[0]}">${pre}</span><span style="color:${color}">${post}</span></span><span style="color:#888;margin-left:10px">${t.meta ? t.meta : formatCount(t.count)}</span></div>`;
            div.style.padding = '4px 8px';
            div.style.cursor = 'pointer';
            div.dataset.tag = t.tag;
            if(i===selected){
                div.style.background = '#333';
            } else {
                div.style.background = '#1e1e1e';
            }
            div.addEventListener('mousedown', (e)=>{
                e.preventDefault();
                insert(area, fragment, t.tag);
            });
            container.appendChild(div);
        });
        const caret = getCaretCoordinates(area, area.selectionStart);
        const rect = area.getBoundingClientRect();
        container.style.left = (window.scrollX + rect.left + caret.left - area.scrollLeft) + 'px';
        container.style.top = (window.scrollY + rect.top + caret.top - area.scrollTop + caret.height) + 'px';
        container.style.width = 'auto';
        container.style.display = 'block';
        selected = -1;
    }

    function insert(area, fragment, tag){
        const cursorPos = area.selectionStart;
        const before = area.value.substring(0, cursorPos);
        const after = area.value.substring(cursorPos);
        const start = before.lastIndexOf(fragment);
        area.value = before.substring(0,start) + tag + after;
        area.selectionStart = area.selectionEnd = start + tag.length;
        container.style.display='none';
        skipInput = true;
        area.dispatchEvent(new Event('input',{bubbles:true}));
    }

    function attach(area){
        createContainer(area);
        area.addEventListener('input', ()=>{
            if(skipInput){ skipInput=false; return; }
            showSuggestions(area);
        });
        area.addEventListener('keydown', (e)=>{
            if(container.style.display==='none') return;
            const items = container.children;
            if(e.key==='ArrowDown'){
                e.preventDefault();
                selected = (selected+1)%items.length;
                updateHighlight(items);
            } else if(e.key==='ArrowUp'){
                e.preventDefault();
                selected = (selected-1+items.length)%items.length;
                updateHighlight(items);
            } else if(e.key==='Enter'){
                if(selected>=0){
                    e.preventDefault();
                    const fragment = area.value.substring(0, area.selectionStart).split(/[,\n]/).pop().trim();
                    insert(area, fragment, items[selected].dataset.tag);
                }
            } else if(e.key==='Tab'){
                e.preventDefault();
                const fragment = area.value.substring(0, area.selectionStart).split(/[,\n]/).pop().trim();
                const idx = selected>=0 ? selected : 0;
                if(items.length>0) insert(area, fragment, items[idx].dataset.tag);
            } else if(e.key==='Escape'){
                container.style.display='none';
            }
        });
    }

    function updateHighlight(items){
        for(let i=0;i<items.length;i++){
            items[i].style.background = (i===selected)?'#333':'#1e1e1e';
        }
    }

    function init(){
        const area = document.querySelector('#positive_prompt textarea');
        if(!area) return;
        loadTags().then(()=>attach(area));
    }

    if(window.onUiLoaded){
        onUiLoaded(init);
    }else{
        window.addEventListener('load', init);
    }
})();
