import csv
import os.path

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

from selenium.webdriver.support.wait import WebDriverWait

def get_data(selectCategory: str):
    url = "https://flk.npc.gov.cn/"
    #设置一些属性配置
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    prefs = {'credentials_enable_service': False, 'profile.password_manager_enabled': False}
    options.add_experimental_option('prefs', prefs)
    options.add_argument('--disable-blink-features=AutomationControlled')

    # 启动Chrome浏览器
    driver = webdriver.Chrome(options=options)
    driver.get('https://flk.npc.gov.cn')
    driver.maximize_window()

    # 根据实际情况定位“法律”选项
    law_element = driver.find_element(By.ID, "flfg")
    law_element.click()

    # 等待页面加载
    time.sleep(3)
    # 点击高级搜索
    advanced_search_element = driver.find_element(By.CSS_SELECTOR, "li.advanced-search")
    advanced_search_element.click()

    #设置保存路径
    file_path = './result'
    # 选择筛选类型
    if selectCategory == "尚未生效":
        checkbox = driver.find_element(By.CSS_SELECTOR, "input.checkboxIpt[name='sxxSearch'][value='3']")
        file_name = os.path.join(file_path,'invalid.csv')
    elif selectCategory == "有效":
        checkbox = driver.find_element(By.CSS_SELECTOR, "input.checkboxIpt[name='sxxSearch'][value='1']")
        file_name = os.path.join(file_path, 'valid.csv')
    elif selectCategory == "已废止":
        checkbox = driver.find_element(By.CSS_SELECTOR, "input.checkboxIpt[name='sxxSearch'][value='9']")
        file_name = os.path.join(file_path, 'delete.csv')
    else:
        checkbox = None
    if checkbox:
        checkbox.click()
        # 等待页面
        time.sleep(1)


    # 进行提交，找到对应法律
    submit = driver.find_element(By.CSS_SELECTOR, ".sx-box .sx-but .sx-butd")
    submit.click()
    print("已进入到: " + selectCategory + "的法律信息")

    # 保存信息到csv文件中
    with open(file_name, 'w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file)
        writer.writerow(['链接', '信息', '制定机关', '法律性质', '时效性', '公布日期'])

        # 设置一个循环，用于遍历抓取数据的页面
        for i in range(2, 101):
            # 查找页面中的所有表格行元素
            infos = driver.find_elements('css selector', '#flData > tr')
            for info in infos:
                # 获取每个表格行中的第一个单元格中的链接属性
                link1 = info.find_element('css selector', '.l-sx .l-wen').get_attribute('onclick')
                # 构造完整的链接地址
                link = "https://flk.npc.gov.cn" + link1.replace("showDetail('.", "").replace("')", "")
                # 打印链接地址
                print(link, end='  ')
                # 获取每个表格行中的第一个单元格中的文本内容
                text = info.find_element('css selector', '.l-sx .l-wen ').text
                # 打印文本内容
                print(text, end='  ')
                # 获取每个表格行中的第二个单元格中的文本内容
                author = info.find_element('css selector', '.l-sx2 .l-wen1 ').text
                # 打印制定机关
                print(author, end='  ')
                # 获取每个表格行中的第三个单元格中的文本内容
                sx3 = info.find_elements('css selector', '.l-sx3 .l-wen1 ')
                root = sx3[0].text
                # 打印法律性质
                print(root)
                timeuse = sx3[1].text
                # 打印法律时效性
                print(timeuse)
                date = info.find_element('css selector', '.l-sx4 .l-wen1').text
                print(date)

                #将数据写入csv文件中
                writer.writerow([link, text, author, root, timeuse, date])
            driver.find_element('css selector', '#layui-laypage-' + str(i) + ' > a.layui-laypage-next').click()
            driver.implicitly_wait(10)
            time.sleep(1)
    print('数据已保存')
    driver.quit()


if __name__ == "__main__":

    selectCategory = "尚未生效"
    get_data(selectCategory)


