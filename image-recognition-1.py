from transformers import pipeline

pipeline = pipeline(task="image-classification", model="Tater86/room-classifier-v1", device="mps")
result = pipeline(images=[
"https://static.dayuse.com/hotels/17502/re-dama-hostel-99092221.jpg",
"https://static.dayuse.com/hotels/17502/re-dama-hostel-99092222.jpg",
"https://static.dayuse.com/hotels/17502/re-dama-hostel-99092223.jpg",
"https://static.dayuse.com/hotels/17502/re-dama-hostel-99092220.jpg",
"https://static.dayuse.com/hotels/17502/re-dama-hostel-99092228.jpg",
"https://static.dayuse.com/hotels/17502/re-dama-hostel-99092231.jpg",
"https://static.dayuse.com/hotels/17502/re-dama-hostel-99092235.jpg"
])
for out in result:
    print(out)
