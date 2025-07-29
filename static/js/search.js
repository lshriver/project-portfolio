/**
 * Search Functionality
 * Provides site-wide search capabilities
 */

document.addEventListener('DOMContentLoaded', function() {
    // Create search elements
    createSearchElements();
    
    // Setup event listeners
    setupSearchListeners();
});

// Create search UI elements
function createSearchElements() {
    // Create search button
    const searchButton = document.createElement('div');
    searchButton.className = 'search-button';
    searchButton.innerHTML = '<span class="search-icon">🔍</span>';
    searchButton.setAttribute('title', 'Search website content');
    
    // Create search modal
    const searchModal = document.createElement('div');
    searchModal.className = 'search-modal';
    searchModal.innerHTML = `
        <div class="search-modal-content">
            <div class="search-modal-close">×</div>
            <h2 class="gradient_text4">Search Content</h2>
            <div class="search-input-container">
                <input type="text" class="search-input" placeholder="Type to search notes, projects, and content...">
                <button class="search-submit">Search</button>
            </div>
            <div class="search-results"></div>
        </div>
    `;
    
    // Append elements to body
    document.body.appendChild(searchButton);
    document.body.appendChild(searchModal);
}

// Setup event listeners for search
function setupSearchListeners() {
    // Search button click
    const searchButton = document.querySelector('.search-button');
    const searchModal = document.querySelector('.search-modal');
    
    if (searchButton && searchModal) {
        searchButton.addEventListener('click', function() {
            searchModal.classList.add('visible');
            setTimeout(() => {
                document.querySelector('.search-input').focus();
            }, 100);
        });
    }
    
    // Close button click
    const closeButton = document.querySelector('.search-modal-close');
    if (closeButton && searchModal) {
        closeButton.addEventListener('click', function() {
            searchModal.classList.remove('visible');
        });
    }
    
    // Close on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && searchModal) {
            searchModal.classList.remove('visible');
        }
    });
    
    // Search input events
    const searchInput = document.querySelector('.search-input');
    const searchSubmit = document.querySelector('.search-submit');
    
    if (searchInput && searchSubmit) {
        // Submit on enter key
        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                performSearch(searchInput.value);
            }
        });
        
        // Submit on button click
        searchSubmit.addEventListener('click', function() {
            performSearch(searchInput.value);
        });
        
        // Live search as typing (debounced)
        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            
            const query = searchInput.value;
            if (query.length >= 3) {
                searchTimeout = setTimeout(() => {
                    performSearch(query);
                }, 300);
            } else if (query.length === 0) {
                clearSearchResults();
            }
        });
    }
}

// Perform search
function performSearch(query) {
    if (!query || query.trim().length < 2) {
        return;
    }
    
    query = query.toLowerCase().trim();
    
    // Show loading in results area
    const resultsContainer = document.querySelector('.search-results');
    if (resultsContainer) {
        resultsContainer.innerHTML = '<div class="search-loading">Searching...</div>';
    }
    
    // Get searchable content from the page
    const searchableContent = getSearchableContent();
    
    // Perform search
    const results = searchContentItems(query, searchableContent);
    
    // Display results
    displaySearchResults(results, query);
}

// Get all searchable content from the current page
function getSearchableContent() {
    const searchableContent = [];
    
    // Search PDF titles and descriptions
    const pdfLinks = document.querySelectorAll('a[href$=".pdf"]');
    pdfLinks.forEach(link => {
        const title = link.textContent;
        const container = link.closest('.note-link');
        let description = '';
        
        if (container) {
            const descElem = container.querySelector('p');
            if (descElem) {
                description = descElem.textContent;
            }
        }
        
        searchableContent.push({
            type: 'pdf',
            title: title,
            description: description,
            url: link.getAttribute('href')
        });
    });
    
    // Search project cards
    const projectCards = document.querySelectorAll('.project-card');
    projectCards.forEach(card => {
        const titleElem = card.querySelector('h3');
        const descElem = card.querySelector('p');
        const linkElem = card.querySelector('a');
        
        if (titleElem && linkElem) {
            searchableContent.push({
                type: 'project',
                title: titleElem.textContent,
                description: descElem ? descElem.textContent : '',
                url: linkElem.getAttribute('href')
            });
        }
    });
    
    // Search headers and paragraphs
    const headers = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    headers.forEach(header => {
        if (!header.closest('.note-link') && !header.closest('.project-card') && !header.closest('.search-modal')) {
            searchableContent.push({
                type: 'content',
                title: header.textContent,
                description: '',
                element: header
            });
        }
    });
    
    const paragraphs = document.querySelectorAll('p');
    paragraphs.forEach(paragraph => {
        if (!paragraph.closest('.note-link') && !paragraph.closest('.project-card') && !paragraph.closest('.search-modal')) {
            searchableContent.push({
                type: 'content',
                title: '',
                description: paragraph.textContent,
                element: paragraph
            });
        }
    });
    
    return searchableContent;
}

// Search content items for query
function searchContentItems(query, items) {
    return items.filter(item => {
        const titleMatch = item.title.toLowerCase().includes(query);
        const descMatch = item.description.toLowerCase().includes(query);
        return titleMatch || descMatch;
    });
}

// Display search results
function displaySearchResults(results, query) {
    const resultsContainer = document.querySelector('.search-results');
    if (!resultsContainer) {
        return;
    }
    
    if (results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="search-no-results">
                <p>No results found for "${query}"</p>
                <p class="search-suggestion">Try a different search term or browse the website using the navigation links.</p>
            </div>
        `;
        return;
    }
    
    let resultsHTML = `<p class="search-results-count">Found ${results.length} result${results.length === 1 ? '' : 's'} for "${query}"</p>`;
    resultsHTML += '<div class="search-results-list">';
    
    results.forEach(result => {
        let resultHTML = '<div class="search-result-item">';
        
        if (result.type === 'pdf') {
            resultHTML += `<div class="search-result-icon">📄</div>`;
            resultHTML += `<div class="search-result-content">
                <a href="${result.url}" target="_blank" class="search-result-title gradient_text4">${result.title}</a>
                <p class="search-result-description">${highlightSearchTerm(result.description, query)}</p>
                <span class="search-result-type">PDF Document</span>
            </div>`;
        } else if (result.type === 'project') {
            resultHTML += `<div class="search-result-icon">🔬</div>`;
            resultHTML += `<div class="search-result-content">
                <a href="${result.url}" class="search-result-title gradient_text1">${result.title}</a>
                <p class="search-result-description">${highlightSearchTerm(result.description, query)}</p>
                <span class="search-result-type">Interactive Project</span>
            </div>`;
        } else if (result.type === 'content') {
            resultHTML += `<div class="search-result-icon">📝</div>`;
            resultHTML += `<div class="search-result-content">`;
            
            if (result.title) {
                resultHTML += `<div class="search-result-title">${highlightSearchTerm(result.title, query)}</div>`;
            }
            
            if (result.description) {
                resultHTML += `<p class="search-result-description">${highlightSearchTerm(result.description, query)}</p>`;
            }
            
            resultHTML += `<span class="search-result-type">Page Content</span>
            </div>`;
        }
        
        resultHTML += '</div>';
        resultsHTML += resultHTML;
    });
    
    resultsHTML += '</div>';
    resultsContainer.innerHTML = resultsHTML;
    
    // Add click event to scroll to content elements
    const contentResults = document.querySelectorAll('.search-result-item');
    contentResults.forEach(result => {
        if (result.querySelector('.search-result-type').textContent === 'Page Content') {
            result.addEventListener('click', function() {
                const resultTitle = result.querySelector('.search-result-title');
                const resultDesc = result.querySelector('.search-result-description');
                const titleText = resultTitle ? resultTitle.textContent : '';
                const descText = resultDesc ? resultDesc.textContent : '';
                
                // Find the element on the page
                const elements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p');
                for (let element of elements) {
                    if ((titleText && element.textContent.includes(titleText)) || 
                        (descText && element.textContent.includes(descText))) {
                        
                        // Close search modal
                        document.querySelector('.search-modal').classList.remove('visible');
                        
                        // Scroll to element
                        setTimeout(() => {
                            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            
                            // Add temporary highlight
                            element.classList.add('search-highlight');
                            setTimeout(() => {
                                element.classList.remove('search-highlight');
                            }, 2000);
                        }, 300);
                        
                        break;
                    }
                }
            });
        }
    });
}

// Highlight search term in text
function highlightSearchTerm(text, term) {
    if (!text) return '';
    
    const regex = new RegExp('(' + term + ')', 'gi');
    return text.replace(regex, '<span class="search-highlight">$1</span>');
}

// Clear search results
function clearSearchResults() {
    const resultsContainer = document.querySelector('.search-results');
    if (resultsContainer) {
        resultsContainer.innerHTML = '';
    }
}

// Export search function for external use
window.searchFunctionality = {
    search: performSearch,
    show: function() {
        const searchModal = document.querySelector('.search-modal');
        if (searchModal) {
            searchModal.classList.add('visible');
            setTimeout(() => {
                document.querySelector('.search-input').focus();
            }, 100);
        }
    }
};