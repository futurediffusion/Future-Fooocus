onUiLoaded(() => {
    const modal = document.getElementById('lightboxModal');
    const img = document.getElementById('modalImage');
    if (!modal || !img) {
        return;
    }

    img.classList.add('zoomable');

    let scale = 1.0;
    let panX = 0;
    let panY = 0;
    let startX = 0;
    let startY = 0;
    let dragging = false;

    function applyTransform() {
        img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }

    function resetTransform() {
        scale = 1.0;
        panX = 0;
        panY = 0;
        img.style.transform = '';
    }

    img.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = img.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const offsetY = e.clientY - rect.top;
        const prevScale = scale;
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        scale = Math.min(5, Math.max(0.2, scale + delta));
        panX += offsetX - offsetX * (scale / prevScale);
        panY += offsetY - offsetY * (scale / prevScale);
        applyTransform();
    });

    img.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        dragging = true;
        startX = e.clientX;
        startY = e.clientY;
        img.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        panX += e.clientX - startX;
        panY += e.clientY - startY;
        startX = e.clientX;
        startY = e.clientY;
        applyTransform();
    });

    document.addEventListener('mouseup', () => {
        dragging = false;
        img.style.cursor = '';
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            resetTransform();
        }
    });

    // reset when modal closes via code
    const originalClose = closeModal;
    closeModal = function() {
        resetTransform();
        originalClose();
    };
});
