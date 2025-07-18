class Item:
    def __init__(self, item_name, category, brand, batch, item_code, price):
        self.item_name = item_name
        self.category = category
        self.brand = brand
        self.batch = batch
        self.item_code = item_code
        self.price = price

    def return_item_details(self):
        return f"Item_name : '{self.item_name}' , category : '{self.category}' , brand : '{self.brand}' , item_code : {self.item_code} , batch : '{self.batch}' , price : {self.price}"

class Shop:
    def __init__(self, shop_name, shop_address, item_list):
        self.shop_name = shop_name
        self.shop_address = shop_address
        self.item_list = item_list
        self.defect_items = []
        self.defect_removed_data = {}

    def add_product(self, item):
        self.item_list.append(item)

    def stock_details(self, category, count):

        counts = {}
        for item in self.item_list:
            if item.category == category:
                counts[item.item_name] = counts.get(item.item_name, 0) + 1


        return [item for item in self.item_list if item.category == category and counts.get(item.item_name, 0) > count]


    def remove_product(self, item_name, item_code, batch):
        self.item_list = [item for item in self.item_list if not (item.item_name == item_name and item.item_code == item_code and item.batch == batch)]

    def add_defect_product(self, item_name, item_code, batch):
        for item in self.item_list:
            if item.item_name == item_name and item.item_code == item_code and item.batch == batch:
                self.defect_items.append(item)
        self.item_list = [item for item in self.item_list if not (item.item_name == item_name and item.item_code == item_code and item.batch == batch)]

    def defect_stock_remove(self, date):
        self.defect_removed_data[date] = self.defect_items
        self.defect_items = []

    def sort_items_by_price(self):
        return sorted(self.item_list, key=lambda item: (item.price, item.item_name))

if __name__ == '__main__':
    n = int(input())
    items = []
    for _ in range(n):
        line = input().replace("'", "")
        item_name, category, brand, batch, item_code, price = line.split(',')
        items.append(Item(item_name, category, brand, batch, int(item_code), int(price)))

    shop_name, shop_address, _ = input().replace("'", "").split(',')
    shop = Shop(shop_name, shop_address, items)

    line = input().replace("'", "")
    item_name, category, brand, batch, item_code, price = line.split(',')
    shop.add_product(Item(item_name, category, brand, batch, int(item_code), int(price)))

    print("Item name Category Brand item_code Price Batch")
    for item in shop.item_list:
        print(f"'{item.item_name}' '{item.category}' '{item.brand}' {item.price} {item.item_code} '{item.batch}'")
    print("---")

    category, count = input().replace("'", "").split(',')
    print(f"Category: {category}, Count: {count}")
    stock = shop.stock_details(category, int(count))
    print(f"Stock: {[item.item_name for item in stock]}")

    line = input().replace("'", "")
    item_name, item_code, batch = line.split(',')
    print(f"Removing: {item_name}, {item_code}, {batch}")
    shop.remove_product(item_name, int(item_code), batch)
    print(f"Items after removal: {[item.item_name for item in shop.item_list]}")

    line = input().replace("'", "")
    item_name, item_code, batch = line.split(',')
    print(f"Adding to defect: {item_name}, {item_code}, {batch}")
    shop.add_defect_product(item_name, int(item_code), batch)
    print(f"Items after adding to defect: {[item.item_name for item in shop.item_list]}")
    print(f"Defect items: {[item.item_name for item in shop.defect_items]}")

    date = input().replace("'", "")
    print(f"Defect stock removal date: {date}")
    shop.defect_stock_remove(date)
    print(f"Defect items after removal: {[item.item_name for item in shop.defect_items]}")
    print(f"Defect removed data: {shop.defect_removed_data}")

    sorted_items = shop.sort_items_by_price()
    print("---")
    for item in sorted_items:
        print(item.return_item_details())
