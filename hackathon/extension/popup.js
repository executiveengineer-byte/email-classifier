document.addEventListener('DOMContentLoaded', function() {
    const serverUrl = 'http://127.0.0.1:5000';

    document.getElementById('open-classifier-dashboard').addEventListener('click', () => {
        chrome.tabs.create({ url: `${serverUrl}/dashboard` });
    });

    document.getElementById('open-sales-dashboard').addEventListener('click', () => {
        chrome.tabs.create({ url: `${serverUrl}/sales_dashboard` });
    });

    document.getElementById('open-send-campaign').addEventListener('click', () => {
        chrome.tabs.create({ url: `${serverUrl}/send_campaign` });
    });
    
    document.getElementById('open-manage-leads').addEventListener('click', () => {
        chrome.tabs.create({ url: `${serverUrl}/leads` });
    });
}); 