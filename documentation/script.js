document.addEventListener('DOMContentLoaded', () => {
    
    /* ===========================
       Dark Mode Logic 
       =========================== */
    const toggleSwitch = document.querySelector('.theme-switch input[type="checkbox"]');
    const architectureImg = document.getElementById('architecture-diagram');
    
    // Check for saved user preference, if any, on load
    const currentTheme = localStorage.getItem('theme');
    
    if (currentTheme) {
        document.documentElement.setAttribute('data-theme', currentTheme);
        if (currentTheme === 'dark') {
            toggleSwitch.checked = true;
            updateArchitectureImage('dark');
        }
    }

    function switchTheme(e) {
        if (e.target.checked) {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            updateArchitectureImage('dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            localStorage.setItem('theme', 'light');
            updateArchitectureImage('light');
        }    
    }

    toggleSwitch.addEventListener('change', switchTheme, false);

    // Helper to switch architecture image if it exists on the page
    function updateArchitectureImage(theme) {
        if (!architectureImg) return;
        
        if (theme === 'dark') {
            architectureImg.src = 'assets/architecture_white.svg';
        } else {
            architectureImg.src = 'assets/architecture.svg';
        }
    }


    /* ===========================
       Smooth Scroll Not Needed for Multi-page 
       (Unless intra-page links are used)
       =========================== */
    
    // Highlight Active Link based on URL
    const currentLocation = location.href;
    const menuItem = document.querySelectorAll('.nav-links a');
    const menuLength = menuItem.length;
    
    for (let i = 0; i < menuLength; i++) {
        if (menuItem[i].href === currentLocation) {
            menuItem[i].className = "active";
        }
    }
});
