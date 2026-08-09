from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('http://localhost:8501')
    page.wait_for_selector('text=Crude Oil')
    elements = page.locator(':text("Crude Oil")').element_handles()
    for el in elements:
        print(el.evaluate('(node) => node.outerHTML'))
    print("--- PARENTS ---")
    el = page.locator(':text("Crude Oil")').first
    print(el.evaluate('(node) => { let s = ""; let n = node; for(let i=0;i<5;i++){ s += n.outerHTML.substring(0, 200) + "\\n"; n = n.parentElement; } return s; }'))
    browser.close()
