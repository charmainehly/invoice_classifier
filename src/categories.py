# kiv
from enum import Enum


class Category(Enum):
    FOOD = 0
    PAYROLL_BENEFITS = 1
    OFFICE_UTILITIES = 2
    M_A = 3
    OFFICE_SUPPLIES = 4
    TRAVEL_TRANSPORT = 5
    PROFESSIONAL_SERVICES = 6
    INSURANCE = 7
    REPAIRS_MAINTENANCE = 8
    SOFTWARE_SUBSCRIPTIONS = 9
    OFFICE_SUPPLIES_2 = 10

    def to_str(self):
        mapping_list = ['Food', 'Payroll & Benefits', 'Office & Utilities', 'Marketing & Advertising',
                        'Office Supplies & Equipment', 'Travel & Transport',
                        'Professional Services', 'Insurance', 'Repairs & Maintenance',
                        'Software & Subscriptions', 'Office Supplies & Equip.']
        return mapping_list[self.value]

    def to_enum(string: str):
        mapping_list = ['Food', 'Payroll & Benefits', 'Office & Utilities', 'Marketing & Advertising',
                        'Office Supplies & Equipment', 'Travel & Transport',
                        'Professional Services', 'Insurance', 'Repairs & Maintenance',
                        'Software & Subscriptions', 'Office Supplies & Equip.']
        
        return Category(mapping_list.index("orange"))


