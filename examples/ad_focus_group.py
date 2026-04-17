# /// script
# dependencies = []
# ///

import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()


model = mf.Model.chat_completion("openai/gpt-4.1-mini")



class AdEvaluation(mf.Signature):
    """Evaluate the advertisement from your perspective. Consider the tone,
    clarity, and appeal. Be honest and specific in your opinion."""

    ad_text: str = mf.InputField(desc="The advertisement text to evaluate")

    opinion: str = mf.OutputField(desc="Your honest reaction to the ad in 2-3 sentences")
    score: int = mf.OutputField(desc="Overall score from 1 (terrible) to 10 (perfect)")



class Teenager(nn.Agent):
    model = model
    system_message = """You are a 17-year-old social media native. You care about
    aesthetics, trends, memes, and authenticity. Focus on whether the ad feels
    authentic or corporate and whether you would share it on social media."""
    signature = AdEvaluation
    message_fields = {"task": {"ad_text": "ad_text"}}
    response_mode = "eval_teenager"


class Professional(nn.Agent):
    model = model
    system_message = """You are a 35-year-old working professional. You value clarity,
    time-saving, and quality. Focus on whether the value proposition is clear
    and whether the ad respects your time."""
    signature = AdEvaluation
    message_fields = {"task": {"ad_text": "ad_text"}}
    response_mode = "eval_professional"


class BudgetShopper(nn.Agent):
    model = model
    system_message = """You are a budget-conscious parent. You look for deals,
    compare prices, and distrust hype. Focus on whether the ad mentions price
    or value and whether it feels honest or manipulative."""
    signature = AdEvaluation
    message_fields = {"task": {"ad_text": "ad_text"}}
    response_mode = "eval_budget"



class CreativeDirector(nn.Agent):
    """Writes the first ad draft from a product description."""

    model = model
    instructions = """
    You are a creative director at an ad agency. Given a product description,
    write compelling ad copy (3-5 sentences). Include:
    - A catchy headline
    - Key benefits
    - A call to action

    Return only the ad text.
    """
    message_fields = {"task": "product_description"}
    response_mode = "ad_text"


class Refiner(nn.Agent):
    """Rewrites ad copy based on focus group feedback."""

    model = model
    instructions = """
    You are a senior copywriter. You receive the original ad text and feedback
    from three customer personas (teenager, professional, budget shopper).

    Rewrite the ad to address their concerns while keeping the core message.
    Return only the new ad text — nothing else.
    """
    message_fields = {"task": "refinement_input"}
    response_mode = "ad_text"



def compute_score(msg):
    scores = [
        msg.eval_teenager.get("score", 5),
        msg.eval_professional.get("score", 5),
        msg.eval_budget.get("score", 5),
    ]
    msg.avg_score = sum(scores) / len(scores)
    msg.iteration = msg.get("iteration", 0) + 1


def prepare_refinement(msg):
    msg.refinement_input = (
        f"Current ad:\n{msg.ad_text}\n\n"
        f"Teenager feedback: {msg.eval_teenager.opinion} "
        f"(score: {msg.eval_teenager.score})\n\n"
        f"Professional feedback: {msg.eval_professional.opinion} "
        f"(score: {msg.eval_professional.score})\n\n"
        f"Budget shopper feedback: {msg.eval_budget.opinion} "
        f"(score: {msg.eval_budget.score})\n\n"
        "Rewrite the ad addressing this feedback."
    )



pipeline = mf.Inline(
    "director"
    " -> [teenager, professional, budget] -> scorer"
    " -> @{avg_score < 8 & iteration < 3}:"
    " prepare -> refiner -> [teenager, professional, budget] -> scorer;",
    {
        "director":     CreativeDirector(),
        "teenager":     Teenager(),
        "professional": Professional(),
        "budget":       BudgetShopper(),
        "refiner":      Refiner(),
        "scorer":       compute_score,
        "prepare":      prepare_refinement,
    },
)


msg = Message()
msg.product_description = """
CloudBrew is a Wi-Fi-enabled coffee maker with a built-in taste profile system.
It learns your preferences over time and adjusts brew strength, temperature,
and grind size automatically. Compatible with any ground coffee or pods.
Retail price: $149. Launch promotion: 40% off pre-orders with free shipping.
"""

pipeline(msg)

print("\n" + "=" * 60)
print("FOCUS GROUP RESULTS")
print("=" * 60)
print(f"\nIterations: {msg.iteration}  |  Final score: {msg.avg_score:.1f}/10")
print(f"\nFinal ad:\n{msg.ad_text}")
print("\n--- Evaluations ---")
for label, field in [
    ("Teenager",       "eval_teenager"),
    ("Professional",   "eval_professional"),
    ("Budget Shopper", "eval_budget"),
]:
    ev = msg.get(field)
    print(f"\n{label} ({ev['score']}/10): {ev['opinion']}")
