import unittest
from inventory import Item, Shop

class TestInventory(unittest.TestCase):
    def test_item(self):
        item = Item('Pencil', 'Stationary', 'ABC', '0123', 1213, 30)
        self.assertEqual(item.return_item_details(), "Item_name : 'Pencil' , category : 'Stationary' , brand : 'ABC' , item_code : 1213 , batch : '0123' , price : 30")

    def test_shop(self):
        item1 = Item('Pencil', 'Stationary', 'ABC', '0123', 1213, 30)
        item2 = Item('Pen', 'Stationary', 'BCD', '0124', 1001, 50)
        shop = Shop('ABC', 'Square Plaza', [item1, item2])
        self.assertEqual(len(shop.item_list), 2)

        shop.add_product(Item('Smartphone', 'Electronics', 'XYZ', '0001', 603, 9000))
        self.assertEqual(len(shop.item_list), 3)

        shop.remove_product('Pencil', 1213, '0123')
        self.assertEqual(len(shop.item_list), 2)

        shop.add_defect_product('Pen', 1001, '0124')
        self.assertEqual(len(shop.item_list), 1)
        self.assertEqual(len(shop.defect_items), 1)

        shop.defect_stock_remove('2020/22/22')
        self.assertEqual(len(shop.defect_items), 0)
        self.assertEqual(len(shop.defect_removed_data), 1)

        item3 = Item('Eraser', 'Stationary', 'DEF', '0125', 1214, 20)
        shop.add_product(item3)
        sorted_items = shop.sort_items_by_price()
        self.assertEqual(sorted_items[0].price, 20)
        self.assertEqual(sorted_items[1].price, 9000)

if __name__ == '__main__':
    unittest.main()
