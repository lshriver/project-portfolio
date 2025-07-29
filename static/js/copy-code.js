/**
 * Copy Code Button Functionality
 * Adds a "Copy" button to all code blocks
 */

document.addEventListener('DOMContentLoaded', function() {
    // Add copy buttons to all code blocks
    addCopyButtons();
});

// Add copy buttons to all code blocks
function addCopyButtons() {
    // Find all pre elements containing code
    const codeBlocks = document.querySelectorAll('pre');
    
    codeBlocks.forEach(block => {
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-code-button';
        copyButton.textContent = 'Copy';
        
        // Style the pre element to accommodate the button
        block.style.position = 'relative';
        
        // Add the button to the code block
        block.appendChild(copyButton);
        
        // Add click event listener
        copyButton.addEventListener('click', function() {
            // Get the text to copy
            const code = block.querySelector('code') || block;
            const text = code.textContent;
            
            // Copy to clipboard
            copyToClipboard(text)
                .then(() => {
                    // Show success feedback
                    copyButton.textContent = 'Copied!';
                    copyButton.classList.add('copied');
                    
                    // Reset button after 2 seconds
                    setTimeout(() => {
                        copyButton.textContent = 'Copy';
                        copyButton.classList.remove('copied');
                    }, 2000);
                })
                .catch(err => {
                    // Show error feedback
                    copyButton.textContent = 'Failed!';
                    copyButton.classList.add('error');
                    
                    // Reset button after 2 seconds
                    setTimeout(() => {
                        copyButton.textContent = 'Copy';
                        copyButton.classList.remove('error');
                    }, 2000);
                    
                    console.error('Copy failed:', err);
                });
        });
    });
}

// Copy text to clipboard
async function copyToClipboard(text) {
    // Use Clipboard API if available
    if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        
        // Make the textarea out of viewport
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        
        textArea.focus();
        textArea.select();
        
        // Copy text
        document.execCommand('copy');
        
        // Clean up
        textArea.remove();
    }
}