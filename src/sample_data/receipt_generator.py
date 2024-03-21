'''
Fake Invoice Generator.
Referenced from: https://github.com/msampathkumar/FakeReceiptImageGenerator
'''

from PIL import Image, ImageDraw, ImageFont, ImageOps

def _insert_text(draw: ImageDraw, x, y, text,
                 color='rgb(0, 0, 0)',
                 font_file='/fonts/Roboto-Bold.ttf',
                 font_size=12):
    text = str(text)
    font = ImageFont.truetype(font_file, size=font_size)
    draw.text((x, y), text, fill=color, font=font)
    return draw


def _combine_all_images_horizantally(images):
    # images = map(Image.open, sys.argv[1:-1])
    w = sum(i.size[0] for i in images)
    mh = max(i.size[1] for i in images)

    result = Image.new("RGBA", (w, mh))

    x = 0
    for i in images:
        result.paste(i, (x, 0))
        x += i.size[0]
    return result


def _combine_all_images_vertically(images):
    # images = map(Image.open, sys.argv[1:-1])
    w = max(i.size[0] for i in images)
    mh = sum(i.size[1] for i in images)

    result = Image.new("RGBA", (w, mh))

    x = 0
    for i in images:
        result.paste(i, (0, x))
        x += i.size[1]
    return result


class ReceiptGenerator():
    def __init__(self, size=None):
        self.header = None
        self.body = None
        self.footer = None
        self.final_output_image = None
        self._debug_ = False  # Debug Param

        self.image_size = size if size else (320, 480)
        self.image_line_sep = self._text_image(
            '--' * 35, font_size=14, size=self.image_size)
        self.image_whitespace_sep = self._text_image(
            ' ', font_size=14, size=self.image_size)

        self.receipt_text_data = []

    def _text_image(self, text, font_size=12, size=(320, 480)):
        width, height = size
        _text = text.center(int(int(width / font_size) * 3.2))
        if self._debug_:
            _text = _text.replace(' ', '.')
        image = Image.new(mode="RGB", size=(
            width, font_size + 4), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw = _insert_text(draw=draw, x=0 + 4, y=0,
                            text=_text, font_size=font_size)
        if self._debug_:
            image = ImageOps.expand(image, border=2, fill='black')
        return image

    def generate_header(self, iter):
        header_text_data = [
            ['Gadget Galleria', '5 Oak St, Binaryburg, TT',
                '104-105-106', 'Receipt: 93670', 'Date: 09/01/2020'],
            ['Matrix Market', '23 2nd St, Techtown, TT',
                '122-123-124', 'Receipt: 10234', 'Date: 11/01/2020'],
            ['Binary Books', '8 Pine Rd, Gadgetville, TT',
                '107-108-109', 'Receipt: 21545', 'Date: 01/01/2021'],
            ['Quantum Quilts', '14 3rd St, Circuit City, TT',
                '113-114-115', 'Receipt: 32856', 'Date: 03/01/2021'],
            ['Prism Provisions', '11 Maple Rd, Techtown, TT',
                '110-111-112', 'Receipt: 44167', 'Date: 05/01/2021'],
            ['Tech Treasures', '9 4th Ave, Code Cove, TT',
                '108-109-110', 'Receipt: 55478', 'Date: 07/01/2021'],
            ['Circuit Ceramics', '27 6th Ave, Gadgetville, TT',
                '126-127-128', 'Receipt: 66789', 'Date: 09/01/2021'],
            ['Gizmo Gifts', '19 5th Ave, Binaryburg, TT',
                '118-119-120', 'Receipt: 78090', 'Date: 11/01/2021'],
            ['Code Coffee', '7 Elm St, Circuit City, TT',
                '106-107-108', 'Receipt: 89401', 'Date: 01/01/2022'],
            ['Widget Workshop', '26 Oak St, Techtown, TT',
                '125-126-127', 'Receipt: 90712', 'Date: 03/01/2022'],
            ['Data Deli', '22 Pine Rd, Gadgetville, TT',
                '121-122-123', 'Receipt: 20123', 'Date: 05/01/2022'],
            ['Byte Bakery', '13 2nd St, Techtown, TT',
                '112-113-114', 'Receipt: 31434', 'Date: 07/01/2022'],
            ['Digital Delights', '16 3rd St, Circuit City, TT',
                '115-116-117', 'Receipt: 42745', 'Date: 09/01/2022'],
            ['Innovate Ice Cream', '4 4th Ave, Code Cove, TT',
                '103-104-105', 'Receipt: 54056', 'Date: 11/01/2022'],
            ['Gadget Galleria', '2 5th Ave, Binaryburg, TT',
                '101-102-103', 'Receipt: 65367', 'Date: 01/01/2023'],
            ['Matrix Market', '30 Elm St, Gadgetville, TT',
                '129-130-131', 'Receipt: 76678', 'Date: 03/01/2023'],
            ['Binary Books', '1 Maple Rd, Techtown, TT',
                '100-101-102', 'Receipt: 87989', 'Date: 05/01/2023'],
            ['Quantum Quilts', '3 Pine Rd, Circuit City, TT',
                '102-103-104', 'Receipt: 99290', 'Date: 07/01/2023'],
            ['Prism Provisions', '25 6th Ave, Gadgetville, TT',
                '124-125-126', 'Receipt: 10501', 'Date: 09/01/2023'],
            ['Tech Treasures', '20 4th Ave, Code Cove, TT',
                '119-120-121', 'Receipt: 21812', 'Date: 11/01/2023'],
            ['Gadget Galleria', '5 Oak St, Binaryburg, TT',
                '104-105-106', 'Receipt: 32901', 'Date: 12/03/2019'],
            ['Matrix Market', '23 2nd St, Techtown, TT',
                '122-123-124', 'Receipt: 43002', 'Date: 02/04/2020'],
            ['Binary Books', '8 Pine Rd, Gadgetville, TT',
                '107-108-109', 'Receipt: 53103', 'Date: 04/05/2021'],
            ['Quantum Quilts', '14 3rd St, Circuit City, TT',
                '113-114-115', 'Receipt: 63204', 'Date: 06/06/2022'],
            ['Prism Provisions', '11 Maple Rd, Techtown, TT',
                '110-111-112', 'Receipt: 73305', 'Date: 08/07/2023'],
            ['Tech Treasures', '9 4th Ave, Code Cove, TT',
                '108-109-110', 'Receipt: 83406', 'Date: 10/08/2019'],
            ['Circuit Ceramics', '27 6th Ave, Gadgetville, TT',
                '126-127-128', 'Receipt: 93507', 'Date: 12/09/2020'],
            ['Gizmo Gifts', '19 5th Ave, Binaryburg, TT',
                '118-119-120', 'Receipt: 103608', 'Date: 02/10/2021'],
            ['Code Coffee', '7 Elm St, Circuit City, TT',
                '106-107-108', 'Receipt: 113709', 'Date: 04/11/2022'],
            ['Widget Workshop', '26 Oak St, Techtown, TT',
                '125-126-127', 'Receipt: 123810', 'Date: 06/12/2023'],
        ]

        image1 = _combine_all_images_vertically([
            self._text_image(
                header_text_data[iter][0], font_size=25, size=(320, 0)),
            self._text_image(
                header_text_data[iter][1], font_size=12, size=(320, 0)),
            self._text_image(
                header_text_data[iter][2], font_size=12, size=(320, 0)),
        ]
        )
        self.receipt_text_data += header_text_data[iter][:3]

        image2 = _combine_all_images_horizantally([
            self._text_image(
                header_text_data[iter][3], font_size=12, size=(160, 0)),
            self._text_image(
                header_text_data[iter][4], font_size=12, size=(160, 0))
        ])
        self.receipt_text_data.append(
            header_text_data[iter][3] + ' ' + header_text_data[iter][4])

        self.header = _combine_all_images_vertically([image1, image2])

    def generate_body(self, iter):
        body_text_data = [
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Chocolate Cake(1 Kg)', '1', '895.00'),
                ('2', 'Flower Bookie', '1', '500.50'),
                ('3', 'Rat Poisen(500ml)', '1', '50.50')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Office Chair', '1', '150.00'),
                ('2', 'Printer Ink Cartridge', '2', '45.00'),
                ('3', 'Stapler', '1', '12.99')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Business Cards (1000)', '1', '75.00'),
                ('2', 'Marketing Brochures (500)', '1', '250.00'),
                ('3', 'Promotional Pens (50)', '2', '20.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Desktop Computer', '1', '800.00'),
                ('2', 'Office Desk', '1', '300.00'),
                ('3', 'Filing Cabinet', '1', '150.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Business Travel Expense', '1', '500.00'),
                ('2', 'Hotel Accommodation', '2', '200.00'),
                ('3', 'Rental Car', '1', '80.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Legal Consultation', '1', '300.00'),
                ('2', 'Accounting Services', '1', '250.00'),
                ('3', 'Web Development', '1', '1000.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Office Insurance Premium', '1', '400.00'),
                ('2', 'Health Insurance Premium', '1', '200.00'),
                ('3', 'Vehicle Insurance Premium', '1', '300.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Office Maintenance Service', '1', '150.00'),
                ('2', 'Computer Repair', '1', '100.00'),
                ('3', 'HVAC Maintenance', '1', '80.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Microsoft Office Subscription', '1', '99.00'),
                ('2', 'Adobe Creative Cloud Subscription', '1', '50.00'),
                ('3', 'Antivirus Software Subscription', '1', '30.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Coffee and Snacks', '1', '50.00'),
                ('2', 'Office Lunch Meeting', '1', '200.00'),
                ('3', 'Catering Service', '1', '300.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Monthly Payroll', '1', '5000.00'),
                ('2', 'Employee Benefits', '1', '1000.00'),
                ('3', 'Overtime Pay', '1', '300.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Electricity Bill', '1', '200.00'),
                ('2', 'Internet Service', '1', '80.00'),
                ('3', 'Water Bill', '1', '50.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Business Trip Airfare', '1', '600.00'),
                ('2', 'Taxi Fare', '1', '50.00'),
                ('3', 'Parking Fees', '1', '20.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'SEO Optimization Service', '1', '400.00'),
                ('2', 'Social Media Marketing', '1', '300.00'),
                ('3', 'Content Writing Service', '1', '200.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Computer Software License', '1', '500.00'),
                ('2', 'Cloud Storage Subscription', '1', '50.00'),
                ('3', 'Project Management Tool Subscription', '1', '30.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Team Building Event Expenses', '1', '1000.00'),
                ('2', 'Office Party Catering', '1', '400.00'),
                ('3', 'Entertainment Tickets', '2', '150.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Employee Health Insurance', '1', '300.00'),
                ('2', 'Dental Insurance', '1', '100.00'),
                ('3', 'Vision Insurance', '1', '80.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Office Cleaning Service', '1', '200.00'),
                ('2', 'Janitorial Supplies', '1', '50.00'),
                ('3', 'Carpet Cleaning', '1', '100.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Conference Registration Fee', '1', '300.00'),
                ('2', 'Speaker Honorarium', '1', '200.00'),
                ('3', 'Exhibition Booth Rental', '1', '500.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Company Car Maintenance', '1', '150.00'),
                ('2', 'Fuel Expense', '1', '100.00'),
                ('3', 'Tire Replacement', '1', '80.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Employee Training Workshop', '1', '500.00'),
                ('2', 'Educational Seminars', '2', '300.00'),
                ('3', 'Training Materials', '1', '150.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Office Renovation', '1', '2000.00'),
                ('2', 'Furniture Upgrade', '1', '800.00'),
                ('3', 'Painting Service', '1', '300.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Business Magazine Subscription', '1', '50.00'),
                ('2', 'Market Research Reports', '1', '200.00'),
                ('3', 'Industry Analysis Tools Subscription', '1', '150.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Employee Rewards Program', '1', '400.00'),
                ('2', 'Incentive Bonuses', '1', '600.00'),
                ('3', 'Recognition Awards', '1', '300.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Gasoline Expense', '1', '100.00'),
                ('2', 'Public Transportation Tickets', '2', '50.00'),
                ('3', 'Taxi Rides', '1', '30.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Employee Health and Wellness Programs', '1', '300.00'),
                ('2', 'Gym Memberships', '2', '100.00'),
                ('3', 'Health Screening Events', '1', '150.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Business Software Upgrade', '1', '500.00'),
                ('2', 'Data Analytics Tools Subscription', '1', '300.00'),
                ('3', 'Cloud Computing Services', '1', '200.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Company Picnic Expenses', '1', '700.00'),
                ('2', 'Team Building Activities', '1', '400.00'),
                ('3', 'Outdoor Catering', '1', '300.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Travel Visa Fees', '1', '100.00'),
                ('2', 'Passport Renewal', '1', '150.00'),
                ('3', 'International Calling Charges', '1', '50.00')
            ],
            [
                ('S.No', 'Item Description', 'Items', 'Cost'),
                ('1', 'Data Security Solutions', '1', '600.00'),
                ('2', 'Firewall Software Subscription', '1', '200.00'),
                ('3', 'Encryption Tools', '1', '150.00')
            ]
        ]

        bag = []

        for each_line in body_text_data[iter]:
            _image_line_item = _combine_all_images_horizantally([
                self._text_image(each_line[0], font_size=12, size=(40, 0)),
                self._text_image(each_line[1], font_size=12, size=(160, 0)),
                self._text_image(each_line[2], font_size=12, size=(40, 0)),
                self._text_image(each_line[3], font_size=12, size=(80, 0)),
            ])
            bag.append(_image_line_item)
            self.receipt_text_data.append('{} {} {} {}'.format(*each_line))

        image3 = _combine_all_images_vertically(bag)

        self.body = image3

    def generate_footer(self):
        footer_text_data = [
            ('      Tax:', '   5.15'),
            ('      GST:', '   1.15'),
            ('      SST:', '   1.15'),
            ('Total Tax:', '1299.10'),
        ]

        bag = []
        for each_line in footer_text_data:
            _image_text = _combine_all_images_horizantally([
                self._text_image('', font_size=14, size=(160, 0)),
                self._text_image(each_line[0], font_size=14, size=(80, 0)),
                self._text_image(each_line[1], font_size=14, size=(80, 0))
            ])
            bag.append(_image_text)
            self.receipt_text_data.append('{} {}'.format(*each_line))

        self.footer = _combine_all_images_vertically(bag)

    def show_output(self):
        pass

    def save_output(self, iter):
        self.generate_header(iter)
        self.generate_body(iter)
        self.generate_footer()
        self.final_output_image = _combine_all_images_vertically(
            [
                self.image_whitespace_sep,
                self.header,
                self.image_whitespace_sep,
                self.image_line_sep,
                self.body,
                self.image_line_sep,
                self.footer,
                self.image_line_sep,
                self.image_whitespace_sep,
            ]
        )
        self.final_output_image.save('./datasets/sample_images/sample_'+str(iter)+'.png')
        print(self.receipt_text_data)


if __name__ == '__main__':
    for iter in range(0, 30):  # 0 - 29
        t = ReceiptGenerator()
        t.save_output(iter)
