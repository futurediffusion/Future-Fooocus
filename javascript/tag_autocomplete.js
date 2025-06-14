// Simple tag autocomplete for the positive prompt textarea
// Loads tags from danbooru.csv and shows suggestions while typing

(function(){
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
        results.forEach((t,i)=>{
            const div = document.createElement('div');
            div.textContent = t;
            div.style.padding = '2px 4px';
            div.style.cursor = 'pointer';
            if(i===selected){
                div.style.background = '#ddd';
            }
            div.addEventListener('mousedown', (e)=>{
                e.preventDefault();
                insert(area, fragment, t);
            });
            container.appendChild(div);
        });
        const rect = area.getBoundingClientRect();
        container.style.left = '0px';
        container.style.top = (area.offsetTop + area.offsetHeight) + 'px';
        container.style.width = rect.width + 'px';
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
                    insert(area, fragment, items[selected].textContent);
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
