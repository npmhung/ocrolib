from PIL import ImageGrab
from xlrd import open_workbook
import os
import win32com.client as win32


def get_sample_from_excel(root_dir='dnp_application_form_final_checked/', out_folder='./excel_ims/', ctype='NUMBER'):
    cur_dir=os.path.abspath(os.path.dirname(__file__))
    total_im=0
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if '$' not in name and ('xlsx' in name.lower() or 'xls' in name.lower()):
                filename = os.path.join(root, name)
                folder = root.split(os.sep)
                folder_name=None
                folder_name=folder[-1]

                print(filename)
                all_labels=[]
                all_row=[]
                wb = open_workbook(filename)
                worksheet = wb.sheet_by_name('By Text')
                num_rows = worksheet.nrows - 1
                num_cells = worksheet.ncols - 1
                curr_row = -1
                while curr_row < num_rows:
                    curr_row += 1
                    row = worksheet.row(curr_row)
                    print('Row: {}'.format(curr_row))
                    curr_cell = -1
                    label = -1
                    while curr_cell < num_cells:
                        curr_cell += 1
                        cell_type = worksheet.cell_type(curr_row, curr_cell)
                        cell_value = worksheet.cell_value(curr_row, curr_cell)
                        print('{}: type {} vs value {}'.format(curr_cell, cell_type, cell_value))
                        if curr_cell == 5:
                            try:
                                if ctype=='NUMBER':
                                    label=str(int(cell_value))
                                else:
                                    label = str(cell_value)

                            except:
                                print('fuck {}'.format(cell_value))
                        if  curr_cell>5 and ctype.lower() in str(cell_value).lower() :
                            all_labels.append(label)
                            all_row.append(curr_row)
                            break
                wb.release_resources()
                del wb
                print(all_labels)
                print(all_row)
                # raise False

                excel = win32.gencache.EnsureDispatch('Excel.Application')
                wb = excel.Workbooks.Open(cur_dir+'/'+filename)
                ws = wb.Worksheets('By Text')
                print(cur_dir+'/'+filename)
                print(ws)
                row=1
                out_folder2 = out_folder + '/' + folder_name
                if not os.path.isdir(out_folder):
                    os.mkdir(out_folder)
                if not os.path.isdir(out_folder2):
                    os.mkdir(out_folder2)

                rchose=0
                for n, shape in enumerate(ws.Shapes):
                    if shape.Name.startswith("Picture"):

                        if row in all_row:
                            # print(shape)
                            try:
                                shape.Copy()
                                im = ImageGrab.grabclipboard()
                                if not im:
                                    shape.Copy()
                                    im = ImageGrab.grabclipboard()
                                if not im:
                                    shape.Copy()
                                    im = ImageGrab.grabclipboard()
                                if not im:
                                    shape.Copy()
                                    im = ImageGrab.grabclipboard()
                                im.save(out_folder2+'/{}-{}.png'.
                                        format(total_im,all_labels[rchose]), 'png')
                            except Exception as e:
                                print('???? {} {} {}'.format(row, str(e), shape))
                            rchose+=1
                            total_im+=1
                        row+=1

                wb.Close()
                excel.Quit()
                # raise False







if __name__=='__main__':
    get_sample_from_excel( out_folder='./excel_ims_kata/', ctype='KATA')
    get_sample_from_excel(root_dir='nissay_combine_report',out_folder='./excel_ims_kata/', ctype='KATA')
