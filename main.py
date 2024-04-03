from tkinter import *

root = Tk() # Создаем корневой объект - окно
root.title('Приложение на Tkinter') # устанавливаем заголовок окна
root.geometry('300x250+1100+400') # устанавливаем размеры окна +400+200 делает смещение окна

root.attributes('-alpha', 0.95) # атрибут задает прозрачность окна

icon = PhotoImage(file="printer_start_icon.png") # устанавливает файл изображения для главного окна
root.iconphoto(False, icon) # задает иконку для окна, первый параметр говорит надо ли ииспользовать иконку по умолчанию для всех окон

root.update_idletasks()

label = Label(text='Hello KillReal.sport') #Создаем текстовую метку
label.pack() #размещаем метку в окне

root.mainloop()