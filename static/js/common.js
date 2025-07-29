// Common JavaScript functionality for the portfolio website

// Initialize tooltips and copy code buttons
document.addEventListener('DOMContentLoaded', function() {
    // Tech icon hover effect
    const techIcons = document.querySelectorAll('.tech-icon');
    
    techIcons.forEach(icon => {
        icon.addEventListener('mouseenter', function() {
            this.querySelector('.tech-icon-tooltip').style.display = 'block';
        });
        
        icon.addEventListener('mouseleave', function() {
            this.querySelector('.tech-icon-tooltip').style.display = 'none';
        });
    });
    
    // General tooltip functionality
    const tooltipContainers = document.querySelectorAll('.tooltip-container');
    
    tooltipContainers.forEach(container => {
        container.addEventListener('mouseenter', function() {
            this.querySelector('.tooltip-content').style.display = 'block';
        });
        
        container.addEventListener('mouseleave', function() {
            this.querySelector('.tooltip-content').style.display = 'none';
        });
    });

    // Copy code button functionality
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Find the code element within the parent code block
            const codeBlock = this.closest('.code-block');
            const codeElement = codeBlock.querySelector('code');
            
            // Get text content
            const codeText = codeElement.textContent;
            
            // Create temporary textarea to copy from
            const textarea = document.createElement('textarea');
            textarea.value = codeText;
            textarea.style.position = 'fixed';  // So it doesn't affect layout
            document.body.appendChild(textarea);
            textarea.select();
            
            // Copy and clean up
            document.execCommand('copy');
            document.body.removeChild(textarea);
            
            // Provide feedback that text was copied
            const originalText = this.textContent;
            this.textContent = 'Copied!';
            this.style.background = 'rgba(34, 247, 147, 0.6)';
            
            // Reset button after 2 seconds
            setTimeout(() => {
                this.textContent = originalText;
                this.style.background = 'rgba(34, 247, 147, 0.2)';
            }, 2000);
        });
    });
});

// Smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const target = document.querySelector(this.getAttribute('href'));
            
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});

// Function to check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}