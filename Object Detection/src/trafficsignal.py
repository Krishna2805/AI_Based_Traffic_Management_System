
#----------------------------------------------------------------------------
import main
import tkinter as tk



root = tk.Tk()
root.title("Green Signal Times for Each Lane")
root.geometry("500x500")    

def calculate_green_time(priority):
    if priority == 0:
        return 10  # Minimum time if no vehicles
    elif priority < 10:
        return 15  # 15 seconds if low priority
    elif priority < 20:
        return 20  # 20 seconds if medium priority
    elif priority < 30: 
        return 25  # 25 seconds if high priority
    else:
        return 30  # 30 seconds if very high priority

for i, priority in enumerate(main.total_priorities_per_image):
    green_time = calculate_green_time(priority)  
    text = f"Lane {i + 1}\nPriority: {priority}\n Green Time: {green_time} seconds"
    
   
    lane_frame = tk.Frame(root, bg="lightgray", width=200, height=100, relief="groove", bd=2)
    lane_frame.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="nsew")
    
    label = tk.Label(lane_frame, text=text, bg="lightgray", font=("Arial", 12), justify="center")
    label.pack(expand=True, fill="both")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()

