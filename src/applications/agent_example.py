import os    
import base64
import requests

from dotenv import load_dotenv
from ..agents.base_agent import BaseAgent

# Load environment variables
load_dotenv()

def main():

    print("Starting agent application...")

    # Create a specialized agent for marketing and RSOC compliance guidelines
    agent = BaseAgent(
        name="MarketingComplianceAgent",
        model_name="gpt-4.1",
        temperature=0.7
    )

    print("Agent created successfully!")

    print("Training agent...")
    
    # Comprehensive training data including RSOC compliance guidelines
    training_data = [
        {
            "input": "What are the key RSOC landing page compliance requirements?",
            "output": "RSOC landing page compliance requirements prohibit: 1) Asking others to click or view ads, 2) Compensating users for viewing ads or searches, 3) Using phrases like 'click the ads' or 'support us', 4) Using arrows or graphics to direct attention to ads, 5) Placing misleading images near ads, 6) Using floating box scripts (sticky ads), 7) Making ads indistinguishable from content, 8) Formatting content to look like ads, 9) Using misleading labels above ad units (only use 'Sponsored Links' or 'Advertisements')."
        },
        {
            "input": "What are the RSOC compliance requirements for referring ads?",
            "output": "RSOC referring ads must NOT: 1) Include specific salary/hourly pay, 2) Falsely offer free services, 3) Claim specific prices, 4) Claim false timelines or effort levels, 5) Include fake option buttons in images, 6) Imply different experiences based on click location, 7) Disguise ads as content, 8) Claim specific discount rates, 9) Use irrelevant images, 10) Promise outcomes after specific clicks, 11) Make health/wellness claims, 12) Claim medical cures, 13) Guarantee approvals, 14) Use inconsistent language across text and image."
        },
        {
            "input": "What phrases and tactics are prohibited in RSOC ads?",
            "output": "Prohibited phrases and tactics include: 1) 'Search' or 'find x', 2) 'Near me', 3) Selling specific items, 4) Clickbait tactics, 5) Sensationalist text/imagery, 6) Direct product/service offers, 7) Phrases like 'our service' or 'let us help', 8) Location-specific terms, 9) Trademark terms implying association, 10) Dynamic keyword insertion, 11) False urgency phrases like 'act now', 12) Absolutes like 'everyone' or 'guaranteed', 13) Specific dollar amounts or percentages."
        },
        {
            "input": "What are the acceptable CTAs for RSOC ad creatives?",
            "output": "The only acceptable CTAs for RSOC ad creatives are: 1) 'Learn More', 2) 'Explore More', 3) 'See More', 4) 'Discover More', 5) 'Find More', 6) 'See Options'. Any other CTAs or variations are not compliant with RSOC guidelines."
        },
        {
            "input": "What are the key principles for RSOC user experience?",
            "output": "Key RSOC user experience principles: 1) Ad-to-landing page experience must be clear and consistent, 2) Advertised content must be realistically fulfillable, 3) Referring ad promise must be fulfilled by content article, 4) Avoid misleading or non-relevant experiences, 5) Content should stand alone without keyword block, 6) User journey from ad to SERP should be transparent, 7) No deceptive implementation methods, 8) No false promises or unrealistic expectations, 9) Maintain consistency across all touchpoints."
        },
        {
            "input": "What are the image and media requirements for RSOC ads?",
            "output": "RSOC image and media requirements: 1) All images/videos must be free use or company-owned, 2) Images must be relevant to ad text, 3) No misleading option buttons or interactive elements, 4) No false UI elements implying different click experiences, 5) No arrows or directional graphics pointing to ads, 6) No misleading imagery alongside ads, 7) No deceptive visual hierarchies, 8) No trademarked imagery without permission, 9) No clickbait-style imagery, 10) No health transformation images making unrealistic claims."
        },
        {
            "input": "What claims are prohibited in RSOC advertising?",
            "output": "Prohibited claims in RSOC advertising include: 1) Specific salary or payment amounts, 2) Guaranteed outcomes or approvals, 3) Specific price points or discounts, 4) Definite timelines or effort levels, 5) Medical or health cure claims, 6) Unrealistic results or transformations, 7) Specific follower or engagement increases, 8) Guaranteed acceptance or approval rates, 9) Location-specific service claims, 10) Claims of direct service provision, 11) Exaggerated performance metrics, 12) Time-sensitive or urgency-based claims."
        },
        {
            "input": "What are the best practices for RSOC ad compliance?",
            "output": "Best practices for RSOC ad compliance: 1) Review all content against current guidelines, 2) Don't assume compliance based on competitors' ads, 3) Maintain consistency between ad and landing page, 4) Use only approved CTAs, 5) Avoid any form of deceptive practices, 6) Keep claims general and realistic, 7) Use only owned or properly licensed media, 8) Ensure transparent user experience, 9) Document compliance checks, 10) Regular audit of ad content and landing pages."
        }
    ]

    # Inject RSOC Landing Page guidelines
    landing_page_init = "What are the restrictions for RSOC landing pages?"
    landing_page_complete = """Here are the RSOC Landing Page restrictions:
1. Cannot ask others to click or view ads or use deceptive implementation methods to obtain clicks or views
2. Cannot compensate users for viewing ads or performing searches or promise compensation to a third party
3. Cannot encourage users to click the ads using phrases like "click the ads", "support us", "visit these links"
4. Cannot direct user attention to the ads using arrows or other graphical gimmicks
5. Cannot place misleading images alongside individual ads
6. Cannot place ads in a floating box script (sticky ads)
7. Cannot format ads so they become indistinguishable from other content
8. Cannot format site content so it's difficult to distinguish from ads
9. Cannot use misleading labels above Google ad units (only "Sponsored Links" or "Advertisements" are allowed)"""
    agent.inject_knowledge(
        init_prompt=landing_page_init,
        complete_prompt=landing_page_complete,
        knowledge_init=landing_page_init,
        knowledge_complete=landing_page_complete
    )

    # Inject RSOC Referring Ads guidelines
    referring_ads_init = "What are the restrictions for RSOC referring ads?"
    referring_ads_complete = """Here are the RSOC Referring Ads restrictions:
1. Cannot include specific salary or hourly pay expectation
2. Cannot falsely offer a free service
3. Cannot claim a product is free or for a specific price
4. Cannot claim a false timeline or level of effort
5. Cannot imply the image includes option buttons
6. Cannot include elements that imply different experiences based on click location
7. Must look like an ad
8. Cannot claim a specific discount rate or percentage
9. Image must be relevant to the ad text
10. Cannot claim user will receive something after specific clicks
11. Cannot make false or misleading health or wellness claims
12. Cannot claim a medical cure or unrealistic outcome
13. Cannot claim specific prices or approval is guaranteed
14. Language must be consistent across text and image
15. Cannot use phrases like 'search', 'find x', 'near me' or sell anything specific
16. Cannot use clickbait tactics or sensationalist text or imagery to drive clicks
17. Cannot directly offer products or services
18. Cannot use phrases like 'our service' or 'let us help' since the article will not provide a product or service to the user"""
    agent.inject_knowledge(
        init_prompt=referring_ads_init,
        complete_prompt=referring_ads_complete,
        knowledge_init=referring_ads_init,
        knowledge_complete=referring_ads_complete
    )

    # Inject General Ad Creation Guidelines
    ad_creation_init = "What are the general guidelines for creating RSOC ads?"
    ad_creation_complete = """Here are the general guidelines for creating RSOC ads:
1. All images or videos must either be free use or owned by the company using the ad
2. Do not misrepresent the ad company as the one fulfilling the product or service
3. Do not assume compliance based on seeing it done by others
4. Do not make exaggerated claims like 'Increase your followers by 50%' or 'Lose weight overnight'
5. Do not include absolutes such as 'everyone', 'no one' or 'guaranteed'
6. Do not use dollar amounts or percentages in ads as you do not control the user outcome
7. Do not create a false sense of urgency like 'act now' or 'supply is limited'
8. Do not include call to action elements to images that misrepresent user experience
9. Do not use dynamic keyword insertion, such as location, as articles aren't dynamically localized
10. Do not use trademark terms (company names) that would mislead users about company association"""
    agent.inject_knowledge(
        init_prompt=ad_creation_init,
        complete_prompt=ad_creation_complete,
        knowledge_init=ad_creation_init,
        knowledge_complete=ad_creation_complete
    )

    # Inject Acceptable CTAs
    cta_init = "What are the acceptable CTAs for RSOC ad creatives?"
    cta_complete = """Here are the only acceptable CTAs for RSOC ad creatives:
1. "Learn More"
2. "Explore More"
3. "See More"
4. "Discover More"
5. "Find More"
6. "See Options"
No other CTAs or variations are allowed."""
    agent.inject_knowledge(
        init_prompt=cta_init,
        complete_prompt=cta_complete,
        knowledge_init=cta_init,
        knowledge_complete=cta_complete
    )

    # Train the agent with RSOC guidelines
    agent.train(training_data)
    
    print("Agent trained successfully!")

    """
    print("Testing agent...")
    # Test the agent with compliance-related questions
    test_questions = [
        "What CTAs can I use in my RSOC ad creative?",
        "How should I handle the user experience from ad to landing page?",
        "What types of claims should I avoid in my ads?",
        "What are the requirements for images in RSOC ads?",
        "How can I ensure my landing page is compliant?",
        "What phrases should I avoid in my ad copy?",
        "Can I use location-specific targeting in my ads?",
        "How should I handle pricing information in ads?"
    ]
    
    print("RSOC Compliance Guidelines Assistant\n" + "="*40)
    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = agent.process_input(question)
        print(f"Response: {response}")
    """

    # Process an actual ad
    ad_primary = "Solar payback may be <5 yrs in many states—read on."
    ad_keywords = {
        "keyword_set_1": "2025 federal solar tax credit explained",
        "keyword_set_2": "average solar panel ROI US",
        "keyword_set_3": "roof requirements for solar install",
        "keyword_set_4": "grid‑tie vs battery backup systems",
        "keyword_set_5": "solar financing options guide"
    }

    # Create a compliance check prompt for the ad
    compliance_check_prompt = f"""
Please evaluate the following ad content for RSOC compliance:

Primary Ad Text:
"{ad_primary}"

Keyword Sets:
1. {ad_keywords['keyword_set_1']}
2. {ad_keywords['keyword_set_2']}
3. {ad_keywords['keyword_set_3']}
4. {ad_keywords['keyword_set_4']}
5. {ad_keywords['keyword_set_5']}

Identify any compliance issues with the primary ad text or keywords based on RSOC guidelines.
List specific violations if any exist, or confirm compliance if no issues are found.
"""

    # Process the compliance check
    print("\nSubmitting ad for compliance check...")
    compliance_response = agent.process_input(compliance_check_prompt)
    print(f"\nCompliance Check Results:\n{compliance_response}")
    
    # Example of extracting text from images
    # TODO: Add image processing

    # Example usage (commented out)
    """
    # Example: Extract text from images
    images = ["path/to/image1.jpg", "path/to/image2.jpg"]
    custom_prompt = "Extract all text from these ad screenshots and check for RSOC compliance."
    extracted_text = extract_text_from_images(images, custom_prompt)
    print(f"\nExtracted Text:\n{extracted_text}")
    
    # Process the extracted text for compliance
    extracted_text_compliance = agent.process_input(
        f"Check if this extracted text from ad images complies with RSOC guidelines:\n\n{extracted_text}"
    )
    print(f"\nCompliance Analysis of Extracted Text:\n{extracted_text_compliance}")
    """
    
    # Save agent state
    agent.save_state() # fix to actually save the state up to 50 latest messages (potentially divided into about 5 prompts)
    print("\nAgent state saved successfully!")

if __name__ == "__main__":
    main() 