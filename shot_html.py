from selenium import webdriver
import time

driver = webdriver.Chrome()

html_file_path = 'file:///D:\Workplace\Python\output\Binnenstad_Rustenburg_Veluvia\Binnenstad_Rustenburg_Veluvia_map_2d.html' 
driver.get(html_file_path)

time.sleep(10)

screenshot_path = 'wageningen_map_lat_lon_2d.png'
driver.save_screenshot(screenshot_path)


driver.quit()

print(f'Screenshot saved to {screenshot_path}')
