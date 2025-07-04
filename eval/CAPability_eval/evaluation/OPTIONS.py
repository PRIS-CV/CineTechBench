












class OPTIONS:
    def __init__(self):
        self.OPTIONS = {
            "Scale": {
                "Extreme Close-Up": (
                    "An Extreme Close-Up (ECU) is a shot that captures a subject in an extremely tight frame, "
                    "focusing on a specific detail of the subject, such as an eye, a mouth, a ring, or a handwritten letter. "
                    "This shot excludes most of the surrounding context, drawing the viewer's attention exclusively to the minute details of the subject."
                ),
                "Close-Up": (
                    "A Close-Up (CU) is a shot that frames the subject's face, head, or a significant object, filling the screen with detailed visual information. "
                    "For human subjects, a Close-Up typically shows the head and shoulders, allowing the audience to focus on facial expressions and emotions."
                ),
                "Medium Close-Up": (
                    "A Medium Close-Up (MCU) is a shot that frames a subject from the chest up, providing a balance between the subject's facial details and body language. "
                    "This shot maintains the emotional focus of the Close-Up while also including some contextual information."
                ),
                "Medium Shot": (
                    "A Medium Shot (MS) frames the subject from the waist up, providing a clear view of both facial expressions and body language. "
                    "It is a versatile shot that strikes a balance between subject focus and contextual surroundings."
                ),
                "Medium Long Shot": (
                    "A Medium Long Shot (MLS), also known as a \"three-quarters shot\", frames the subject from the knees up, providing a broader view of the subject within the setting. "
                    "It is often used to maintain a sense of the subject's body language while still focusing on the individual."
                ),
                "Long Shot": (
                    "A Long Shot (LS) is a wide framing that captures the entire subject from head to toe, along with a significant portion of the surrounding environment. "
                    "The subject is visible but occupies a relatively smaller portion of the frame."
                ),
                "Extreme Long Shot": (
                    "An Extreme Long Shot (ELS), also known as a Wide Shot (WS) or Establishing Shot, captures a vast expanse of the setting, "
                    "with the subject appearing very small or even insignificant within the environment. This shot may cover vast landscapes, cityscapes, or wide action scenes."
                ),
            },
            "Angle": {
                "High Angle Shot": (
                    "A High Angle Shot is captured with the camera positioned above the subject, angled downward. "
                    "This perspective often makes the subject appear smaller, weaker, or vulnerable, depending on the narrative context."
                ),
                "Low Angle Shot": (
                    "A Low Angle Shot is captured with the camera positioned below the subject, angled upward. "
                    "This perspective makes the subject appear larger, more dominant, or intimidating."
                ),
                "Bird's Eye View": (
                    "A Bird's Eye View (or Overhead Shot) is an extremely high angle shot taken directly above the subject, providing a top-down perspective. "
                    "This view emphasizes spatial layout and geometric patterns within the scene."
                ),
                "Worm's Eye View": (
                    "A Worm's Eye View is an extreme low-angle shot taken from below the subject, almost directly upwards. "
                    "This perspective can make subjects appear overwhelmingly large or powerful, or it can capture towering structures from ground level."
                ),
                "Diagonal Angle": (
                    "A Diagonal Angle is a camera angle that captures the subject from a non-frontal or backside, non-profile perspective. "
                    "The camera is positioned at an intermediate angle between the subject's side and front or back, typically ranging from approximately 30° to 60° off-axis. "
                    "This versatile angle allows the viewer to perceive multiple dimensions of the subject simultaneously, offering a more dynamic and three-dimensional representation."
                ),
                "Profile Shot": (
                    "A Profile Shot is captured with the camera positioned to the side of the subject, showing the subject's profile or side view. "
                    "This framing emphasizes the subject's silhouette, facial contours, and gestures."
                ),
                "Back Shot": (
                    "A Back Shot is a camera angle taken from behind the subject, typically showing the subject's back or shoulders while they face away from the camera. "
                    "This can also include over-the-shoulder shots."
                ),
            },
            "Composition": {
                "Symmetrical": (
                    "Symmetrical Composition is a compositional technique where elements within the frame are arranged in a balanced and mirror-like manner, "
                    "creating a sense of harmony and equilibrium. This can be achieved through vertical, horizontal, or radial symmetry."
                ),
                "Central": (
                    "Central Composition is a technique where the main subject is positioned at the exact center of the frame, drawing immediate attention to it. "
                    "This approach uses the inherent strength of central focus, often resulting in a powerful and direct visual impact."
                ),
                "Diagonal": (
                    "Diagonal Composition is a compositional technique that uses diagonal lines or elements within the frame to guide the viewer's eye and create a sense of movement, "
                    "depth, and dynamism. These diagonal lines can be naturally present in the scene (such as a sloping path, a leaning tree, or a crossing shadow) "
                    "or can be intentionally created by tilting the camera (known as a Dutch Angle or Tilted Composition). This approach allows for a dramatic and visually engaging effect."
                ),
                "Rule of Thirds": (
                    "The Rule of Thirds is a compositional guideline that divides the frame into nine equal sections with two horizontal and two vertical lines. "
                    "The main subjects are placed along these lines or at their intersections, creating a balanced and naturally pleasing composition."
                ),
                "Framing": (
                    "Framing is a technique where elements within the scene are used to naturally frame the subject, directing the viewer’s focus towards it. "
                    "These framing elements can include natural objects (such as trees), architectural elements (such as windows), or other elements within the environment."
                ),
                "Curved Line": (
                    "Curved Line Composition uses naturally occurring or deliberately arranged curved lines within the frame to guide the viewer’s eye, "
                    "create a sense of flow, or emphasize the softness of the scene. These lines can be literal (such as a winding road) or implied (such as a subject’s pose)."
                ),
                "Horizontal": (
                    "Horizontal Composition is a technique where the main visual elements are arranged along a horizontal axis, emphasizing width and creating a sense of stability. "
                    "This can be achieved using the horizon line, landscapes, or other horizontally aligned subjects."
                ),
            },
            "Colors": {
                "Red": (
                    "Red is a warm, highly intense color often associated with strong emotions, including passion, love, anger, danger, and urgency. "
                    "In cinematography, it is used to draw attention, create tension, or symbolize strong emotional states."
                ),
                "Yellow": (
                    "Yellow is a bright, warm color that is often associated with happiness, optimism, energy, and warmth. However, it can also represent caution, anxiety, or deceit, depending on the context."
                ),
                "Blue": (
                    "Blue is a cool, calming color commonly associated with tranquility, stability, melancholy, and introspection. "
                    "It is widely used to convey a sense of calmness, sadness, or detachment."
                ),
                "Green": (
                    "Green is a color often associated with nature, growth, freshness, and harmony. However, in certain contexts, it can also represent envy, corruption, or toxicity."
                ),
                "Purple": (
                    "Purple is a color traditionally associated with royalty, luxury, mystery, and spirituality. "
                    "It can evoke both sophistication and fantasy, depending on the context."
                ),
                "Black and White": (
                    "Black and White is a monochrome color scheme that removes all hues, focusing on contrasts between light and dark. "
                    "This style emphasizes texture, composition, lighting, and shadow, often creating a timeless, dramatic, or nostalgic aesthetic."
                ),
            },
            "Lighting": {
                "High Key": (
                    "High Key Lighting is a lighting technique characterized by bright, even illumination with minimal shadows and a high level of ambient light. "
                    "This style is achieved using multiple light sources or a large, soft light source to reduce contrast."
                ),
                "Low Key": (
                    "Low Key Lighting is a dramatic lighting technique that emphasizes strong contrast between light and dark areas, with deep shadows and minimal fill light. "
                    "It is achieved using a primary light source with little to no fill light."
                ),
                "Hard Light": (
                    "Hard Light is a type of lighting that produces sharp, well-defined shadows and high contrast between illuminated and dark areas. "
                    "It is created using a small, direct light source such as a spotlight or bare bulb."
                ),
                "Soft Light": (
                    "Soft Light is a lighting technique that produces diffused, gentle illumination with gradual transitions between light and shadow. "
                    "This effect is achieved using large light sources, diffusion panels, softboxes, or indirect lighting."
                ),
                "Back Light": (
                    "Back Light is a lighting technique where the light source is positioned behind the subject, often creating a rim or halo effect around the subject’s outline. "
                    "This light separates the subject from the background and adds depth to the scene."
                ),
                "Side Light": (
                    "Side Light is a lighting technique where the light source is placed at a 90-degree angle to the subject, illuminating one side while leaving the other side in shadow. "
                    "This creates a strong contrast between light and darkness."
                ),
                "Top Light": (
                    "Top Light is a lighting technique where the light source is placed directly above the subject, casting shadows downward. "
                    "This creates dramatic shadows on the subject’s face and emphasizes the upper contours."
                ),
            },
            "Focal Lengths": {
                "Standard Lens": (
                    "A Standard Lens, also known as a Normal Lens, is a lens with a focal length that closely matches the human eye's natural field of view. "
                    "In most cases, this ranges between 35mm to 50mm for full-frame cameras. Standard lenses provide a balanced perspective without significant distortion, making them highly versatile for various types of scenes."
                ),
                "Medium Focal Length": (
                    "Medium Focal Length refers to lenses with a focal length slightly longer than standard lenses, typically between 50mm and 85mm for full-frame cameras. "
                    "These lenses offer moderate compression and a slightly narrowed field of view, making subjects appear closer without the extreme effects of telephoto lenses."
                ),
                "Telephoto Lens": (
                    "A Telephoto Lens is a long-focus lens with a focal length greater than 85mm, typically ranging from 85mm to 300mm or beyond for full-frame cameras. "
                    "These lenses provide a narrow field of view and significant background compression, making distant subjects appear closer."
                ),
                "Fisheye Lens": (
                    "A Fisheye Lens is an ultra-wide-angle lens with a focal length typically between 8mm and 16mm, designed to capture an extremely wide field of view, often with a 180° angle. "
                    "It creates a distinctive curved, distorted image, which can be either circular (full-frame fisheye) or rectangular (rectilinear fisheye)."
                ),
                "Macro Lengs": (
                    "A Macro Lens is a specialized lens designed for extreme close-up photography, capable of achieving a high level of magnification (typically 1:1 or greater). "
                    "These lenses have a short minimum focusing distance, allowing detailed capture of small subjects."
                ),
            },
        }