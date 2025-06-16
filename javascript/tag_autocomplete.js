// Enhanced tag autocomplete for Stable Diffusion WebUI
// Improved performance, better UX, and cleaner code architecture

(function() {
    'use strict';

    // Configuration with better defaults
    const CONFIG = {
        tagFiles: ['a1111-sd-webui-tagcomplete/tags/danbooru.csv', 'a1111-sd-webui-tagcomplete/tags/extra-quality-tags.csv'],
        chantFiles: ['a1111-sd-webui-tagcomplete/tags/demo-chants.json'],
        maxResults: 8,
        minQueryLength: 2,
        enabled: true,
        appendComma: true,
        appendSpace: true,
        replaceUnderscores: false,
        escapeParentheses: false,
        debounceMs: 150,
        fuzzySearch: false,
        ...window.tac_user_config
    };

    // State management
    const state = {
        tags: new Map(), // Use Map for O(1) lookups
        chants: [],
        containers: new Map(),
        selectedIndex: -1,
        currentTextarea: null,
        debounceTimer: null,
        cache: new Map() // Cache search results
    };

    // Utility functions
    const utils = {
        debounce(func, wait) {
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(state.debounceTimer);
                    func(...args);
                };
                clearTimeout(state.debounceTimer);
                state.debounceTimer = setTimeout(later, wait);
            };
        },

        formatCount(num) {
            if (!num) return '';
            return new Intl.NumberFormat('en', {
                notation: 'compact',
                maximumFractionDigits: 1
            }).format(num);
        },

        parseCSV(line) {
            const result = [];
            let current = '';
            let inQuotes = false;
            
            for (let i = 0; i < line.length; i++) {
                const char = line[i];
                if (char === '"') {
                    inQuotes = !inQuotes;
                } else if (char === ',' && !inQuotes) {
                    result.push(current.trim());
                    current = '';
                } else {
                    current += char;
                }
            }
            result.push(current.trim());
            return result;
        },

        getCaretPosition(element) {
            const selection = window.getSelection();
            const range = selection.getRangeAt(0);
            const preCaretRange = range.cloneRange();
            preCaretRange.selectNodeContents(element);
            preCaretRange.setEnd(range.endContainer, range.endOffset);
            return preCaretRange.toString().length;
        },

        // Improved caret coordinates calculation
        getCaretCoordinates(element, position) {
            const properties = [
                'direction', 'boxSizing', 'width', 'height', 'overflowX', 'overflowY',
                'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth',
                'borderStyle', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
                'fontStyle', 'fontVariant', 'fontWeight', 'fontStretch', 'fontSize',
                'fontSizeAdjust', 'lineHeight', 'fontFamily', 'textAlign', 'textTransform',
                'textIndent', 'textDecoration', 'letterSpacing', 'wordSpacing'
            ];

            const div = document.createElement('div');
            const span = document.createElement('span');
            
            Object.assign(div.style, {
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word',
                position: 'absolute',
                visibility: 'hidden',
                overflow: 'hidden'
            });

            const computed = getComputedStyle(element);
            properties.forEach(prop => {
                div.style[prop] = computed[prop];
            });

            div.textContent = element.value.substring(0, position);
            span.textContent = element.value.substring(position) || '.';
            div.appendChild(span);
            document.body.appendChild(div);

            const coordinates = {
                top: span.offsetTop + parseInt(computed.borderTopWidth),
                left: span.offsetLeft + parseInt(computed.borderLeftWidth),
                height: parseInt(computed.lineHeight)
            };

            document.body.removeChild(div);
            return coordinates;
        },

        // Fuzzy search implementation
        fuzzyMatch(query, target) {
            if (!CONFIG.fuzzySearch) return target.toLowerCase().includes(query.toLowerCase());
            
            const queryLower = query.toLowerCase();
            const targetLower = target.toLowerCase();
            let queryIndex = 0;
            
            for (let i = 0; i < targetLower.length && queryIndex < queryLower.length; i++) {
                if (targetLower[i] === queryLower[queryIndex]) {
                    queryIndex++;
                }
            }
            
            return queryIndex === queryLower.length;
        }
    };

    // Data loading with better error handling and caching
    const dataLoader = {
        async loadTags() {
            const loadedTags = new Map();
            
            for (const file of CONFIG.tagFiles) {
                try {
                    const response = await fetch(`file=${file}`);
                    if (!response.ok) {
                        console.warn(`Failed to load tag file: ${file}`);
                        continue;
                    }
                    
                    const text = await response.text();
                    const lines = text.split('\n').filter(line => line.trim());
                    
                    for (const line of lines) {
                        const parts = utils.parseCSV(line);
                        if (parts.length >= 1 && parts[0]) {
                            const tag = parts[0].trim();
                            if (!loadedTags.has(tag)) {
                                const entry = { tag };
                                
                                if (parts.length >= 3) {
                                    const count = parseInt(parts[2]);
                                    if (!isNaN(count)) {
                                        entry.count = count;
                                    } else {
                                        entry.meta = parts[2];
                                    }
                                }
                                
                                if (!entry.meta && parts.length >= 2) {
                                    const maybeCount = parseInt(parts[1]);
                                    if (isNaN(maybeCount)) {
                                        entry.meta = parts[1];
                                    }
                                }
                                
                                loadedTags.set(tag, entry);
                            }
                        }
                    }
                } catch (error) {
                    console.error(`Error loading tag file ${file}:`, error);
                }
            }
            
            state.tags = loadedTags;
            console.log(`Loaded ${loadedTags.size} tags`);
        },

        async loadChants() {
            const loadedChants = [];
            
            for (const file of CONFIG.chantFiles) {
                try {
                    const response = await fetch(`file=${file}`);
                    if (!response.ok) {
                        console.warn(`Failed to load chant file: ${file}`);
                        continue;
                    }
                    
                    const json = await response.json();
                    if (Array.isArray(json)) {
                        json.forEach(chant => {
                            if (chant?.name && chant?.content) {
                                loadedChants.push(chant);
                            }
                        });
                    }
                } catch (error) {
                    console.error(`Error loading chant file ${file}:`, error);
                }
            }
            
            state.chants = loadedChants;
            console.log(`Loaded ${loadedChants.length} chants`);
        }
    };

    // UI components
    const ui = {
        createContainer() {
            const container = document.createElement('div');
            Object.assign(container.style, {
                position: 'absolute',
                background: '#1e1e1e',
                color: '#fff',
                border: '1px solid #444',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.7)',
                zIndex: '10000',
                display: 'none',
                minWidth: '200px',
                maxHeight: '300px',
                overflowY: 'auto',
                fontSize: '14px',
                fontFamily: 'monospace'
            });
            
            document.body.appendChild(container);
            return container;
        },

        createSuggestionItem(content, isSelected = false, isChant = false) {
            const div = document.createElement('div');
            Object.assign(div.style, {
                padding: '6px 12px',
                cursor: 'pointer',
                background: isSelected ? '#333' : '#1e1e1e',
                borderBottom: '1px solid #333',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'background-color 0.15s ease'
            });

            if (isChant) {
                div.innerHTML = `<span style="color: #4fc3f7">${content.name}</span>`;
                div.dataset.chant = content.content;
            } else {
                const colors = ['#ffa500', '#f48fb1', '#f06292', '#ec407a', '#e91e63'];
                const randomColor = colors[Math.floor(Math.random() * colors.length)];
                
                div.innerHTML = `
                    <span style="color: ${randomColor}">${content.tag}</span>
                    <span style="color: #888; font-size: 12px">
                        ${content.meta || utils.formatCount(content.count)}
                    </span>
                `;
                div.dataset.tag = content.tag;
            }

            // Hover effects
            div.addEventListener('mouseenter', () => {
                if (!isSelected) div.style.background = '#2a2a2a';
            });
            
            div.addEventListener('mouseleave', () => {
                if (!isSelected) div.style.background = '#1e1e1e';
            });

            return div;
        },

        positionContainer(container, textarea) {
            const caret = utils.getCaretCoordinates(textarea, textarea.selectionStart);
            const rect = textarea.getBoundingClientRect();
            
            const left = window.scrollX + rect.left + caret.left - textarea.scrollLeft;
            const top = window.scrollY + rect.top + caret.top - textarea.scrollTop + caret.height;
            
            // Ensure container stays within viewport
            const maxLeft = window.innerWidth - container.offsetWidth - 10;
            const maxTop = window.innerHeight - container.offsetHeight - 10;
            
            container.style.left = Math.min(left, maxLeft) + 'px';
            container.style.top = Math.min(top, maxTop) + 'px';
        }
    };

    // Search functionality
    const search = {
        isChantFragment(fragment) {
            const lower = fragment.toLowerCase();
            return lower.startsWith('<') && 
                   !lower.startsWith('<e:') && 
                   !lower.startsWith('<h:') && 
                   !lower.startsWith('<l:') &&
                   !fragment.includes(' ');
        },

        searchTags(query) {
            const cacheKey = `tags_${query}`;
            if (state.cache.has(cacheKey)) {
                return state.cache.get(cacheKey);
            }

            const results = [];
            const queryLower = query.toLowerCase();
            
            for (const [tag, data] of state.tags) {
                if (utils.fuzzyMatch(query, tag)) {
                    results.push(data);
                }
                if (results.length >= CONFIG.maxResults * 2) break; // Get more for better sorting
            }

            // Enhanced sorting
            results.sort((a, b) => {
                const aStartsWith = a.tag.toLowerCase().startsWith(queryLower);
                const bStartsWith = b.tag.toLowerCase().startsWith(queryLower);
                
                if (aStartsWith && !bStartsWith) return -1;
                if (!aStartsWith && bStartsWith) return 1;
                
                const aQuality = a.meta?.toLowerCase().includes('quality') ?? false;
                const bQuality = b.meta?.toLowerCase().includes('quality') ?? false;
                
                if (aQuality && !bQuality) return -1;
                if (!aQuality && bQuality) return 1;
                
                return (b.count || 0) - (a.count || 0);
            });

            const limited = results.slice(0, CONFIG.maxResults);
            state.cache.set(cacheKey, limited);
            return limited;
        },

        searchChants(query) {
            const cacheKey = `chants_${query}`;
            if (state.cache.has(cacheKey)) {
                return state.cache.get(cacheKey);
            }

            const searchTerm = query.toLowerCase()
                .replace('<chant:', '')
                .replace('<c:', '')
                .replace('<', '');

            const results = state.chants.filter(chant => 
                chant.name?.toLowerCase().includes(searchTerm) ||
                chant.terms?.toLowerCase().includes(searchTerm)
            ).slice(0, CONFIG.maxResults);

            state.cache.set(cacheKey, results);
            return results;
        }
    };

    // Suggestion display
    const suggestions = {
        show(textarea, fragment) {
            if (fragment.length < CONFIG.minQueryLength) {
                this.hide(textarea);
                return;
            }

            const container = state.containers.get(textarea);
            const isChant = search.isChantFragment(fragment);
            const results = isChant ? search.searchChants(fragment) : search.searchTags(fragment);

            if (results.length === 0) {
                this.hide(textarea);
                return;
            }

            container.innerHTML = '';
            state.selectedIndex = -1;

            results.forEach((result, index) => {
                const item = ui.createSuggestionItem(result, false, isChant);
                
                item.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    this.insert(textarea, fragment, isChant ? result.content : result.tag, isChant);
                });

                container.appendChild(item);
            });

            ui.positionContainer(container, textarea);
            container.style.display = 'block';
            state.currentTextarea = textarea;
        },

        hide(textarea) {
            const container = state.containers.get(textarea);
            if (container) {
                container.style.display = 'none';
            }
            state.selectedIndex = -1;
            state.currentTextarea = null;
        },

        insert(textarea, fragment, replacement, isChant = false) {
            const cursorPos = textarea.selectionStart;
            const before = textarea.value.substring(0, cursorPos);
            const after = textarea.value.substring(cursorPos);
            const startPos = before.lastIndexOf(fragment);

            if (startPos === -1) return;

            let insertion = replacement;
            
            if (!isChant) {
                if (CONFIG.replaceUnderscores) {
                    insertion = insertion.replace(/_/g, ' ');
                }
                if (CONFIG.escapeParentheses) {
                    insertion = insertion.replace(/\(/g, '\\(').replace(/\)/g, '\\)');
                }
                if (CONFIG.appendComma) insertion += ',';
            }
            
            if (CONFIG.appendSpace) insertion += ' ';

            textarea.value = before.substring(0, startPos) + insertion + after;
            textarea.selectionStart = textarea.selectionEnd = startPos + insertion.length;
            
            this.hide(textarea);
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
        },

        updateSelection(direction) {
            if (!state.currentTextarea) return;
            
            const container = state.containers.get(state.currentTextarea);
            const items = container.children;
            
            if (items.length === 0) return;

            // Clear previous selection
            if (state.selectedIndex >= 0) {
                items[state.selectedIndex].style.background = '#1e1e1e';
            }

            // Update selection
            if (direction === 'down') {
                state.selectedIndex = (state.selectedIndex + 1) % items.length;
            } else if (direction === 'up') {
                state.selectedIndex = state.selectedIndex <= 0 ? items.length - 1 : state.selectedIndex - 1;
            }

            // Highlight new selection
            items[state.selectedIndex].style.background = '#333';
            items[state.selectedIndex].scrollIntoView({ block: 'nearest' });
        }
    };

    // Event handlers
    const handlers = {
        debouncedShowSuggestions: utils.debounce((textarea) => {
            const cursorPos = textarea.selectionStart;
            const text = textarea.value.substring(0, cursorPos);
            const fragment = text.split(/[,\n]/).pop().trim();
            
            if (fragment.length === 0) {
                suggestions.hide(textarea);
                return;
            }
            
            suggestions.show(textarea, fragment);
        }, CONFIG.debounceMs),

        onInput(textarea) {
            return () => {
                this.debouncedShowSuggestions(textarea);
            };
        },

        onKeyDown(textarea) {
            return (e) => {
                const container = state.containers.get(textarea);
                if (!container || container.style.display === 'none') return;

                const items = container.children;
                let handled = false;

                switch (e.key) {
                    case 'ArrowDown':
                        suggestions.updateSelection('down');
                        handled = true;
                        break;
                    case 'ArrowUp':
                        suggestions.updateSelection('up');
                        handled = true;
                        break;
                    case 'Enter':
                    case 'Tab':
                        if (state.selectedIndex >= 0 && items[state.selectedIndex]) {
                            const item = items[state.selectedIndex];
                            const fragment = textarea.value.substring(0, textarea.selectionStart)
                                .split(/[,\n]/).pop().trim();
                            
                            if (item.dataset.chant) {
                                suggestions.insert(textarea, fragment, item.dataset.chant, true);
                            } else if (item.dataset.tag) {
                                suggestions.insert(textarea, fragment, item.dataset.tag, false);
                            }
                        }
                        handled = true;
                        break;
                    case 'Escape':
                        suggestions.hide(textarea);
                        handled = true;
                        break;
                }

                if (handled) {
                    e.preventDefault();
                    e.stopPropagation();
                }
            };
        },

        onBlur(textarea) {
            return () => {
                // Hide suggestions after a short delay to allow clicks
                setTimeout(() => suggestions.hide(textarea), 150);
            };
        }
    };

    // Main attachment function
    function attachToTextarea(textarea) {
        if (state.containers.has(textarea)) return; // Already attached

        const container = ui.createContainer();
        state.containers.set(textarea, container);

        textarea.addEventListener('input', handlers.onInput(textarea));
        textarea.addEventListener('keydown', handlers.onKeyDown(textarea));
        textarea.addEventListener('blur', handlers.onBlur(textarea));

        // Cleanup on textarea removal
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.removedNodes.forEach((node) => {
                    if (node === textarea) {
                        if (container.parentNode) {
                            container.parentNode.removeChild(container);
                        }
                        state.containers.delete(textarea);
                        observer.disconnect();
                    }
                });
            });
        });

        observer.observe(document.body, { childList: true, subtree: true });
    }

    // Initialization
    async function initialize() {
        if (!CONFIG.enabled) return;

        try {
            console.log('Loading tag autocomplete data...');
            await Promise.all([
                dataLoader.loadTags(),
                dataLoader.loadChants()
            ]);

            // Attach to existing textareas
            const selectors = [
                '#positive_prompt textarea',
                '#negative_prompt textarea',
                'textarea[placeholder*="prompt"]'
            ];

            selectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(attachToTextarea);
            });

            // Watch for dynamically added textareas
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            selectors.forEach(selector => {
                                if (node.matches && node.matches(selector)) {
                                    attachToTextarea(node);
                                } else {
                                    node.querySelectorAll?.(selector).forEach(attachToTextarea);
                                }
                            });
                        }
                    });
                });
            });

            observer.observe(document.body, { childList: true, subtree: true });
            
            console.log('Tag autocomplete initialized successfully');
        } catch (error) {
            console.error('Failed to initialize tag autocomplete:', error);
        }
    }

    // Start initialization
    if (typeof onUiLoaded === 'function') {
        onUiLoaded(initialize);
    } else {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initialize);
        } else {
            initialize();
        }
    }

    // Expose API for external configuration
    window.tagAutocomplete = {
        config: CONFIG,
        state: state,
        reload: initialize,
        clearCache: () => state.cache.clear()
    };

})();
