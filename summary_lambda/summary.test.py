from summary import handler
import json

# Properly format the body with a dictionary before dumping to JSON
event = {
    "body": json.dumps({
        "text": '''My trip to Paris felt like stepping into a living canvas, where every street corner whispered stories and every sunrise painted the city anew. From the moment I arrived, the rhythm of the city wrapped around me—cobblestone streets echoing with footsteps, cafés humming with quiet conversation, and the scent of fresh croissants drifting through the air. I wandered without agenda, letting curiosity guide me through Montmartre’s artistic alleys and along the Seine’s poetic banks.
The Eiffel Tower, shimmering at night, was more than a landmark—it was a reminder of how beauty can be both grand and intimate. I spent hours at the Louvre, not just admiring the art but absorbing the silence between brushstrokes. In Le Marais, I found vintage bookstores and falafel that rivaled any Michelin meal. Paris taught me that elegance lives in the details: the curve of a wrought-iron balcony, the flicker of candlelight in a tucked-away bistro, the way locals greet each other with warmth and ease.
One afternoon, I sat in Jardin du Luxembourg, watching children sail toy boats across the pond. It was a moment of stillness that felt like eternity. Travel often reveals who we are when we’re not trying to be anything—and Paris, with its timeless grace, invited me to simply be.
I left with a heart fuller than my suitcase, carrying memories stitched with laughter, awe, and quiet reflection. Paris wasn’t just a destination—it was a feeling I’ll return to again and again.'''
    }),
    "queryStringParameters": {
        "points": "3"
    }
}

response = handler(event, {})
print(response)