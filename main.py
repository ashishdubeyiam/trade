class Item:
    def __init__(self, item_name, category, brand, batch, item_code, price):
        self.item_name = item_name
        self.category = category
        self.brand = brand
        self.batch = batch
        self.item_code = item_code
        self.price = price

    def return_item_details(self):
        return f"Item_name : {self.item_name} , category : {self.category} , brand : {self.brand} , item_code : {self.item_code} , batch : {self.batch} , price : {self.price}"

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
        category_items = [item for item in self.item_list if item.category == category]

        counts = {}
        for item in category_items:
            counts[item.item_name] = counts.get(item.item_name, 0) + 1

        return [item for item in category_items if counts[item.item_name] > count]


    def remove_product(self, item_name, item_code, batch):
        item_to_remove = None
        for item in self.item_list:
            if item.item_name == item_name and item.item_code == item_code and item.batch == batch:
                item_to_remove = item
                break
        if item_to_remove:
            self.item_list.remove(item_to_remove)

    def add_defect_product(self, item_name, item_code, batch):
        item_to_move = None
        for item in self.item_list:
            if item.item_name == item_name and item.item_code == item_code and item.batch == batch:
                item_to_move = item
                break
        if item_to_move:
            self.defect_items.append(item_to_move)
            self.item_list.remove(item_to_move)

    def defect_stock_remove(self, date):
        self.defect_removed_data[date] = self.defect_items
        self.defect_items = []

    def sort_items_by_price(self):
        return sorted(self.item_list, key=lambda item: (item.price, item.item_name))

if __name__ == '__main__':
    n = int(input())
    items = []
    for _ in range(n):
        item_name, category, brand, batch, item_code, price = input().split(',')
        items.append(Item(item_name, category, brand, batch, int(item_code), int(price)))

    shop_name, shop_address, _ = input().split(',')
    shop = Shop(shop_name, shop_address, items)

    item_name, category, brand, batch, item_code, price = input().split(',')
    shop.add_product(Item(item_name, category, brand, batch, int(item_code), int(price)))

    print("Item name Category Brand item_code Price Batch")
    for item in shop.item_list:
        print(f"{item.item_name} {item.category} {item.brand} {item.price} {item.item_code} {item.batch}")

    category, count = input().split(',')
    stock = shop.stock_details(category, int(count))

    item_name, item_code, batch = input().split(',')
    shop.remove_product(item_name, int(item_code), batch)

    item_name, item_code, batch = input().split(',')
    shop.add_defect_product(item_name, int(item_code), batch)

    date = input()
    shop.defect_stock_remove(date)

    sorted_items = shop.sort_items_by_price()
    for item in sorted_items:
        print(item.return_item_details())
