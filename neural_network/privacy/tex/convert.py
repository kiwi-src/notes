import pdf2image
import os

pdf2image.convert_from_path('model.pdf', dpi=300, fmt='png',
                            output_folder=".", output_file='model.png', transparent=True)

os.rename('model.png0001-1.png', 'model.png')
