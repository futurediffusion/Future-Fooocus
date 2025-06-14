// Simple tag autocomplete for the positive prompt textarea
// Loads tags from danbooru.csv and shows suggestions while typing

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
    const TAG_PATH = 'file=a1111-sd-webui-tagcomplete/tags/danbooru.csv';
    let tags = [];
    let container; // suggestion container
    let selected = -1;

    async function loadTags(){
        try {
            const resp = await fetch(TAG_PATH);
            if(!resp.ok) return;
            const text = await resp.text();
            tags = text.split(/\n/).map(line => line.split(',')[0]);
        } catch(err){
            console.error('Failed to load tags', err);
        }
    }

    function createContainer(area){
        container = document.createElement('div');
        container.style.position = 'absolute';
        container.style.background = '#fff';
        container.style.border = '1px solid #ccc';
        container.style.zIndex = 1000;
        container.style.display = 'none';
        container.style.maxHeight = '200px';
        container.style.overflowY = 'auto';
        container.style.borderRadius = '8px';
        container.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        container.style.minWidth = '150px';
        area.parentElement.style.position = 'relative';
        area.parentElement.appendChild(container);
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
        const results = tags.filter(t => t.startsWith(lower)).slice(0,10);
        if(results.length === 0){
            container.style.display = 'none';
            return;
        }
        container.innerHTML = '';
        const colors = ['#ffa500','#f48fb1','#f06292','#ec407a','#e91e63'];
        results.forEach((t,i)=>{
            const div = document.createElement('div');
            const pre = t.substring(0, fragment.length);
            const post = t.substring(fragment.length);
            const color = colors[(i+1)%colors.length];
            div.innerHTML = `<span style="color:${colors[0]}">${pre}</span><span style="color:${color}">${post}</span>`;
            div.style.padding = '2px 4px';
            div.style.cursor = 'pointer';
            div.dataset.tag = t;
            if(i===selected){
                div.style.background = '#ddd';
            }
            div.addEventListener('mousedown', (e)=>{
                e.preventDefault();
                insert(area, fragment, t);
            });
            container.appendChild(div);
        });
        const caret = getCaretCoordinates(area, area.selectionStart);
        container.style.left = (caret.left - area.scrollLeft) + 'px';
        container.style.top = (caret.top - area.scrollTop + caret.height) + 'px';
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
        area.dispatchEvent(new Event('input',{bubbles:true}));
    }

    function attach(area){
        createContainer(area);
        area.addEventListener('input', ()=>showSuggestions(area));
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
            } else if(e.key==='Escape'){
                container.style.display='none';
            }
        });
    }

    function updateHighlight(items){
        for(let i=0;i<items.length;i++){
            items[i].style.background = (i===selected)?'#ddd':'#fff';
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
