window.main_viewer_height = 512;

function refresh_grid() {
    let gridContainer = document.querySelector('#final_gallery .grid-container');
    let final_gallery = document.getElementById('final_gallery');

    if (gridContainer) if (final_gallery) {
        let rect = final_gallery.getBoundingClientRect();
        let cols = Math.ceil((rect.width - 16.0) / rect.height);
        if (cols < 2) cols = 2;
        gridContainer.style.setProperty('--grid-cols', cols);
    }
}

function refresh_grid_delayed() {
    refresh_grid();
    setTimeout(refresh_grid, 100);
    setTimeout(refresh_grid, 500);
    setTimeout(refresh_grid, 1000);
}

function resized() {
    let windowHeight = window.innerHeight - 260;
    let elements = document.getElementsByClassName('main_view');

    if (windowHeight > 745) windowHeight = 745;

    for (let i = 0; i < elements.length; i++) {
        elements[i].style.height = windowHeight + 'px';
    }

    window.main_viewer_height = windowHeight;

    refresh_grid();
}

function viewer_to_top(delay = 100) {
    setTimeout(() => window.scrollTo({top: 0, behavior: 'smooth'}), delay);
}

function viewer_to_bottom(delay = 100) {
    let element = document.getElementById('positive_prompt');
    let yPos = window.main_viewer_height;

    if (element) {
        yPos = element.getBoundingClientRect().top + window.scrollY;
    }

    setTimeout(() => window.scrollTo({top: yPos - 8, behavior: 'smooth'}), delay);
}

window.addEventListener('resize', (e) => {
    resized();
});

onUiLoaded(async () => {
    resized();
});

function on_style_selection_blur() {
    let target = document.querySelector("#gradio_receiver_style_selections textarea");
    target.value = "on_style_selection_blur " + Math.random();
    let e = new Event("input", {bubbles: true})
    Object.defineProperty(e, "target", {value: target})
    target.dispatchEvent(e);
}

onUiLoaded(async () => {
    let spans = document.querySelectorAll('.aspect_ratios span');

    spans.forEach(function (span) {
        span.innerHTML = span.innerHTML.replace(/&lt;/g, '<').replace(/&gt;/g, '>');
    });

    let aspectContainer = document.querySelector('.aspect_ratios');
    if (aspectContainer && !aspectContainer.querySelector('.aspect_ratios_group')) {
        let labels = Array.from(aspectContainer.querySelectorAll('label'));
        aspectContainer.innerHTML = '';
        const groups = [
            {title: 'Square', count: 2, cls: 'square-ratio'},
            {title: 'Portrait', count: 4, cls: 'portrait-ratio'},
            {title: 'Landscape', count: labels.length - 6, cls: 'landscape-ratio'}
        ];
        let idx = 0;
        groups.forEach(g => {
            let grp = document.createElement('div');
            grp.classList.add('aspect_ratios_group');
            let title = document.createElement('div');
            title.textContent = g.title;
            title.style.fontWeight = 'bold';
            title.style.width = '100%';
            grp.appendChild(title);
            for (let i = 0; i < g.count; i++) {
                if (labels[idx]) {
                    let label = labels[idx];
                    let span = label.querySelector('span');
                    if (span) {
                        let text = span.textContent;
                        span.textContent = '';
                        span.classList.add(g.cls);
                        label.insertBefore(document.createTextNode(text), span);
                    }
                    grp.appendChild(label);
                } else {
                    let blank = document.createElement('div');
                    blank.style.flex = '0 0 calc(50% - 5px)';
                    grp.appendChild(blank);
                }
                idx++;
            }
            aspectContainer.appendChild(grp);
        });
    }

    let styleSelections = document.querySelector('.style_selections');
    if (styleSelections) {
        styleSelections.addEventListener('focusout', function (event) {
            setTimeout(() => {
                if (!this.contains(document.activeElement)) {
                    on_style_selection_blur();
                }
            }, 200);
        });
    }

    let inputs = document.querySelectorAll('.lora_weight input[type="range"]');

    inputs.forEach(function (input) {
        input.style.marginTop = '12px';
    });
});
