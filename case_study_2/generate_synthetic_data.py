import json
import random
import uuid

def generate_events(outcome):
    events = [{"timestamp": 0, "type": "call_start"}]
    current_ts = random.randint(1, 3)
    
    # number of dialogue turns
    if outcome == "abandoned":
        num_turns = random.randint(1, 4)
    elif outcome == "error":
        num_turns = random.randint(2, 6)
    else:
        num_turns = random.randint(4, 15)
        
    for _ in range(num_turns):
        # agent speaks
        duration_ms = random.randint(1000, 10000)
        words = duration_ms // 300
        events.append({
            "timestamp": current_ts,
            "type": "speech_detected",
            "data": {
                "speaker": "agent",
                "duration_ms": duration_ms,
                "words": words
            }
        })
        current_ts += (duration_ms // 1000) + random.randint(1, 3)
        
        # user speaks
        duration_ms = random.randint(500, 8000)
        words = duration_ms // 300
        events.append({
            "timestamp": current_ts,
            "type": "speech_detected",
            "data": {
                "speaker": "user",
                "duration_ms": duration_ms,
                "words": words
            }
        })
        current_ts += (duration_ms // 1000) + random.randint(1, 3)
            
        # maybe a tool call
        if random.random() < 0.15:
            tools = ["submit_survey_response", "check_calendar", "verify_insurance", "fetch_patient_record"]
            events.append({
                "timestamp": current_ts,
                "type": "tool_call",
                "data": {
                    "tool": random.choice(tools)
                }
            })
            current_ts += random.randint(1, 5)

    events.append({"timestamp": current_ts, "type": "call_end"})
    return events


def generate_mock_data(num_samples=500):
    agents = ["receptionist", "triage_nurse", "billing_specialist", "support_agent", "scheduler"]
    orgs = ["org_a", "org_b", "org_c", "org_d"]
    purposes = ["sdoh_screening", "appointment_scheduling", "billing_inquiry", "prescription_refill", "general_inquiry"]
    phone_types = ["mobile", "landline", "voip"]
    times = ["morning", "afternoon", "evening", "night"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    outcomes = ["completed", "abandoned", "transferred", "error"]
    
    calls = []
    
    for i in range(1, num_samples + 1):
        outcome = random.choices(outcomes, weights=[0.5, 0.25, 0.15, 0.1])[0]
        
        survey_completion_rate = random.random() if outcome == "completed" else 0.0
        
        call = {
            "call_id": str(uuid.uuid4()),
            "metadata": {
                "agent_id": random.choice(agents),
                "org_id": random.choice(orgs),
                "call_purpose": random.choice(purposes),
                "caller_phone_type": random.choice(phone_types),
                "time_of_day": random.choice(times),
                "day_of_week": random.choice(days)
            },
            "events": generate_events(outcome),
            "outcome": outcome,
            "survey_completion_rate": round(survey_completion_rate, 2)
        }
        
        calls.append(call)
        
    return {"calls": calls}


if __name__ == "__main__":
    data = generate_mock_data(501)
    with open("data/calls.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Generated 550 synthetic calls and saved to calls.json")
