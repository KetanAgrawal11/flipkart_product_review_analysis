import csv
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os

# === CONFIGURATION ===
CHROMEDRIVER_PATH = r"C:\Users\Ketan Agrawal\Downloads\chromedriver-win64\chromedriver.exe"
URL = "https://www.flipkart.com/samsung-9-kg-5-star-ai-ecobubble-super-speed-wi-fi-hygiene-steam-digital-inverter-motor-fully-automatic-front-load-washing-machine-in-built-heater-grey/product-reviews/itm6c8a617aef39c?pid=WMNH7SPNGXDGUKVE&lid=LSTWMNH7SPNGXDGUKVEZAHO0B&marketplace=FLIPKART"
OUTPUT_FILE = os.path.join("flipkart_product-review-analysis\data", "raw", "Samsung_washing_machine_reviews.csv")
TARGET_COUNT = 500

# === BROWSER SETUP ===
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--lang=en-US,en;q=0.9")
options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
# options.add_argument("--headless")

# Automatically download the correct ChromeDriver version
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def scrape_flipkart_reviews():
    print("Opening Flipkart reviews page...")
    driver.get(URL)

    WebDriverWait(driver, 25).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.ZmyHeo"))
    )
    time.sleep(2)

    reviews_data = []
    page = 1

    while len(reviews_data) < TARGET_COUNT:
        print(f"\nScraping Page {page}... ({len(reviews_data)}/{TARGET_COUNT})")
        time.sleep(random.uniform(2, 4))

        # Re-fetch review blocks and buttons on the current page
        review_blocks = driver.find_elements(By.CSS_SELECTOR, "div.col.EPCmJX.Ma1fCG")  # per-card wrapper
        read_more_buttons = driver.find_elements(By.CSS_SELECTOR, "div.col.EPCmJX.Ma1fCG span.b4x-fr")

        # Click all "Read More" buttons
        for btn in read_more_buttons:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.2)
            except Exception:
                continue

        # Re-fetch elements again after DOM update
        review_blocks = driver.find_elements(By.CSS_SELECTOR, "div.col.EPCmJX.Ma1fCG")

        count_this_page = 0
        for i in range(len(review_blocks)):
            if len(reviews_data) >= TARGET_COUNT:
                break
            try:
                block = review_blocks[i]

                # rating: <div class="XQDdHH ...">5<img ...>
                try:
                    rating = block.find_element(By.CSS_SELECTOR, "div.XQDdHH").text.strip()
                except Exception:
                    rating = ""

                # title: <p class="z9E0IG">The best in the Range! Go for it...</p>
                try:
                    title = block.find_element(By.CSS_SELECTOR, "p.z9E0IG").text.strip()
                except Exception:
                    title = ""

                # review text: <div class="ZmyHeo"><div><div class=""> ... </div></div></div>
                try:
                    review_el = block.find_element(By.CSS_SELECTOR, "div.ZmyHeo > div > div")
                    review = (review_el.get_attribute("innerText") or review_el.text).replace("READ MORE", "").strip()
                except Exception:
                    review = ""

                if review:
                    reviews_data.append([rating, title, review])
                    count_this_page += 1
            except Exception:
                continue

        print(f"Collected {count_this_page} reviews from page {page}.")

        # Navigate to next page
        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
            )
            driver.execute_script("arguments[0].scrollIntoView();", next_button)
            driver.execute_script("arguments[0].click();", next_button)
            page += 1
            time.sleep(random.uniform(2, 4))  # wait for page load
        except Exception:
            print("No more pages found or Next button missing.")
            break

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rating", "Title", "Review"])
        writer.writerows(reviews_data)


    print(f"\nDone! Scraped {len(reviews_data)} total reviews.")
    print(f"Saved to {OUTPUT_FILE}")
    driver.quit()

if __name__ == "__main__":
    scrape_flipkart_reviews()
