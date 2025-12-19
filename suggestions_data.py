# suggestions_data.py
# Kho dữ liệu gợi ý tìm kiếm cho CLIP Image Search

SUGGESTIONS_DB = {
    "🐕 Dog Lovers": [
        "golden retriever puppy", "husky with blue eyes", "german shepherd standing", 
        "cute shiba inu", "corgi running on grass", "bulldog sleeping", 
        "poodle haircut", "beagle sniffing", "labrador swimming", "dog wearing glasses",
        "border collie agility", "dalmatian spots", "dachshund long body",
        "dog playing fetch", "fluffy samoyed", "chihuahua barking", "dog at beach"
    ],
    "🦁 Animal Kingdom": [
        "ginger cat stretching", "siamese cat portrait", "black panther hunting",
        "lion roaring in savannah", "elephant family walking", "giraffe eating leaves",
        "red fox in snow", "polar bear on ice", "giant panda bamboo", 
        "brown bear fishing", "wolf howling moon", "rabbit hopping",
        "majestic tiger", "monkey playing", "deer in misty forest"
    ],
    "🌊 Sea & Sky": [
        "blue whale jumping", "dolphin swimming", "great white shark", 
        "clownfish in anemone", "sea turtle underwater", "jellyfish glowing",
        "eagle flying high", "colorful peacock feathers", "owl in hollow tree",
        "pink flamingo standing", "kingfisher diving", "penguin on iceberg"
    ],
    "🏔️ Nature Wonders": [
        "mount everest peak", "grand canyon sunset", "amazon rainforest drone",
        "sahara desert dunes", "niagara waterfalls", "northern lights aurora",
        "volcano eruption lava", "lavender fields provence", "cherry blossom japan",
        "autumn maple forest", "tropical island aerial", "misty pine mountains",
        "rainbow after rain", "thunderstorm lightning", "sunny flower meadow"
    ],
    "🗼 Travel & Cities": [
        "tokyo shibuya crossing", "new york times square", "paris eiffel tower night",
        "london big ben", "venice canals gondola", "santorini white houses",
        "great wall of china", "dubai burj khalifa", "rome colosseum",
        "abandoned castle", "modern skyscraper glass", "cozy snowy village"
    ],
    "🍕 Food & Lifestyle": [
        "delicious beef steak", "sushi rolls platter", "italian pasta spaghetti",
        "pancakes with syrup", "fresh fruit smoothie", "hot chocolate marshmallow",
        "wine glass sunset", "baked croissant", "healthy avocado toast",
        "yoga on beach", "hiking in mountains", "reading book by fire",
        "camping under stars", "surfing big wave"
    ],
    "🎨 Styles & Moods": [
        "cinematic lighting", "minimalist white aesthetic", "dark moody vibes",
        "vintage film photography", "black and white portrait", "macro insect detail",
        "bokeh background city", "cyberpunk neon lights", "street photography",
        "abstract oil painting", "flat lay workspace", "aerial drone view",
        "long exposure stars", "pastel color palette"
    ]
}

# Tạo danh sách tất cả keywords để tìm kiếm nhanh
ALL_KEYWORDS = [item for sublist in SUGGESTIONS_DB.values() for item in sublist]