import msgflux as mf
import msgflux.nn as nn

msg = mf.Message(
    content="Describe the image and identify any sounds.",
    images={"horse": "https://example.com/horse.jpg"},
    audios={"dog": "https://example.com/dog.mp3"}
)

fake_agent = nn.Agent("teste", fake_model)

fake_agent._set_task_inputs("content")
fake_agent._set_task_multimodal_inputs({
    "images": ["images.horse"],
    #"audios": ["audios.dog"]
})


fake_agent._set_task_template("The capital of {{country}} is {{city}}")
fake_agent._prepare_task({"country": "France", "city": "Paris"})