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
        return [item for item in self.item_list if item.category == category and self.item_list.count(item) > count]

    def remove_product(self, item_name, item_code, batch):
        self.item_list = [item for item in self.item_list if not (item.item_name == item_name and item.item_code == item_code and item.batch == batch)]

    def add_defect_product(self, item_name, item_code, batch):
        for item in self.item_list:
            if item.item_name == item_name and item.item_code == item_code and item.batch == batch:
                self.defect_items.append(item)
                self.item_list.remove(item)
                break

    def defect_stock_remove(self, date):
        self.defect_removed_data[date] = self.defect_items
        self.defect_items = []

    def sort_items_by_price(self):
        return sorted(self.item_list, key=lambda item: item.price)
