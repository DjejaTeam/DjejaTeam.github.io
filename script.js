// Mobile Navigation Toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-menu a').forEach(link => {
    link.addEventListener('click', () => {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    });
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Animate elements on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-on-scroll');
        }
    });
}, observerOptions);

// Observe sections for animation
document.querySelectorAll('section').forEach(section => {
    observer.observe(section);
});

// Counter animation for impact stats
function animateCounter(element, target, duration = 2000) {
    let start = 0;
    const increment = target / (duration / 16);
    
    const timer = setInterval(() => {
        start += increment;
        element.textContent = Math.floor(start);
        
        if (start >= target) {
            element.textContent = target;
            clearInterval(timer);
        }
    }, 16);
}

// Trigger counter animation when impact section is visible
const impactSection = document.querySelector('.impact');
const counters = document.querySelectorAll('.stat-number');
let countersAnimated = false;

const impactObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !countersAnimated) {
            countersAnimated = true;
            counters.forEach(counter => {
                const target = parseInt(counter.textContent);
                counter.textContent = '0';
                animateCounter(counter, target);
            });
        }
    });
}, { threshold: 0.5 });

if (impactSection) {
    impactObserver.observe(impactSection);
}

// Newsletter signup functionality
const newsletterForm = document.querySelector('.newsletter-signup');
const emailInput = newsletterForm?.querySelector('input[type="email"]');
const subscribeBtn = newsletterForm?.querySelector('button');

if (newsletterForm) {
    subscribeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const email = emailInput.value.trim();
        
        if (email && isValidEmail(email)) {
            // Simulate API call
            subscribeBtn.textContent = 'Subscribing...';
            subscribeBtn.disabled = true;
            
            setTimeout(() => {
                subscribeBtn.textContent = 'Subscribed!';
                subscribeBtn.style.background = '#28a745';
                emailInput.value = '';
                
                setTimeout(() => {
                    subscribeBtn.textContent = 'Subscribe';
                    subscribeBtn.style.background = '';
                    subscribeBtn.disabled = false;
                }, 2000);
            }, 1000);
        } else {
            emailInput.style.borderColor = '#dc3545';
            emailInput.placeholder = 'Please enter a valid email';
            
            setTimeout(() => {
                emailInput.style.borderColor = '';
                emailInput.placeholder = 'Email address';
            }, 3000);
        }
    });
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Header background change on scroll
const navbar = document.querySelector('.navbar');
let lastScrollY = window.scrollY;

window.addEventListener('scroll', () => {
    const currentScrollY = window.scrollY;
    
    if (currentScrollY > 100) {
        navbar.style.background = 'rgba(139, 156, 103, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(139, 156, 103, 0.95)';
        navbar.style.boxShadow = 'none';
    }
    
    // Hide/show navbar on scroll
    if (currentScrollY > lastScrollY && currentScrollY > 200) {
        navbar.style.transform = 'translateY(-100%)';
    } else {
        navbar.style.transform = 'translateY(0)';
    }
    
    lastScrollY = currentScrollY;
});

// Add transition to navbar
navbar.style.transition = 'all 0.3s ease';

// Parallax effect for hero section
const hero = document.querySelector('.hero');
const heroContent = document.querySelector('.hero-content');

window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const rate = scrolled * -0.5;
    
    if (hero && scrolled < hero.offsetHeight) {
        heroContent.style.transform = `translateY(${rate}px)`;
    }
});

// Add loading animation
window.addEventListener('load', () => {
    document.body.classList.add('loaded');
});

// Create floating elements animation
function createFloatingElements() {
    const hero = document.querySelector('.hero');
    if (!hero) return;
    
    for (let i = 0; i < 6; i++) {
        const star = document.createElement('div');
        star.className = 'floating-star';
        star.style.cssText = `
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 50%;
            animation: float ${3 + Math.random() * 4}s ease-in-out infinite;
            animation-delay: ${Math.random() * 2}s;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
        `;
        hero.appendChild(star);
    }
}

// Add CSS for floating animation
const style = document.createElement('style');
style.textContent = `
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.6; }
        50% { transform: translateY(-20px) rotate(180deg); opacity: 1; }
    }
    
    .loaded {
        opacity: 1;
    }
    
    body:not(.loaded) {
        opacity: 0;
        transition: opacity 0.5s ease;
    }
`;
document.head.appendChild(style);

// Initialize floating elements
createFloatingElements();

// Button hover effects
document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-2px) scale(1.05)';
    });
    
    btn.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// Loop diagram animation
const loopItems = document.querySelectorAll('.loop-item');
let currentItem = 0;

function animateLoop() {
    loopItems.forEach((item, index) => {
        item.style.opacity = index === currentItem ? '1' : '0.6';
        item.style.transform = index === currentItem ? 'scale(1.1)' : 'scale(1)';
    });
    
    currentItem = (currentItem + 1) % loopItems.length;
}

// Start loop animation when visible
const loopObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const interval = setInterval(animateLoop, 1500);
            // Store interval to clear it later if needed
            entry.target.setAttribute('data-interval', interval);
        } else {
            const interval = entry.target.getAttribute('data-interval');
            if (interval) {
                clearInterval(interval);
            }
        }
    });
}, { threshold: 0.5 });

const djejaLoop = document.querySelector('.djeja-loop');
if (djejaLoop) {
    loopObserver.observe(djejaLoop);
}
