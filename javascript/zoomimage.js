onUiLoaded(() => {
    const modal = document.getElementById('lightboxModal');
    const img = document.getElementById('modalImage');
    if (!modal || !img) {
        return;
    }

    img.classList.add('zoomable');
    img.style.transformOrigin = '0 0';
    img.draggable = false;
    img.addEventListener('dragstart', (e) => e.preventDefault());

    let scale = 1.0;
    let panX = 0;
    let panY = 0;
    let startX = 0;
    let startY = 0;
    let dragging = false;
    let moved = false;

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
        const imgX = (e.clientX - rect.left - panX) / scale;
        const imgY = (e.clientY - rect.top - panY) / scale;
        const prevScale = scale;
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        const newScale = Math.min(5, Math.max(0.2, scale + delta));
        panX -= imgX * (newScale - prevScale);
        panY -= imgY * (newScale - prevScale);
        scale = newScale;
        applyTransform();
    });

    img.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        dragging = true;
        moved = false;
        startX = e.clientX;
        startY = e.clientY;
        img.style.cursor = 'grabbing';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
            moved = true;
        }
        panX += dx;
        panY += dy;
        startX = e.clientX;
        startY = e.clientY;
        applyTransform();
    });

    function stopDrag() {
        dragging = false;
        img.style.cursor = '';
    }

    document.addEventListener('mouseup', stopDrag);
    img.addEventListener('mouseup', stopDrag);
    modal.addEventListener('mouseleave', stopDrag);

    img.onclick = (e) => {
        if (moved) {
            e.stopPropagation();
            moved = false;
        } else {
            closeModal();
        }
    };

    modal.onclick = (e) => {
        if (moved) {
            e.stopPropagation();
            moved = false;
        } else if (e.target === modal) {
            closeModal();
        }
    };

    // reset when modal closes via code
    const originalClose = closeModal;
    closeModal = function() {
        resetTransform();
        originalClose();
    };
});
