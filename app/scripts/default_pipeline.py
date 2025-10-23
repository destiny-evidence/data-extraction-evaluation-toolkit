"""Pseudocode exploration of what our typical pipeline might look like."""

from app.data_models.pipeline import Pipeline, PipelineStage, Job
from app.processors.parser import ParserLibrary


default_pipeline =  Pipeline(stages = [parse, translate_eppi, extract, write_stats])

default_pipeline.execute()


default_pipeline.extend(PipelineStage('my_script.R'), job_type='validation') # add at the end
default_pipeline.insert(PipelineStage(type='CODE'), index=1)

default_pipeline.analyse() # benchmarking, performance analysis

default_pipeline.replace(index=0, my_new_parser)

# class Job(BaseModel):
#     """The attributes describing a specific job."""

#     name: str
#     job_format: JobFormat
#     job_type: JobType | list[JobType]
#     language: Language
#     ingress_method: IngressMethod | None  # we may have a job that starts with no data
#     egress_method: EgressMethod



my_new_parser = ParserLibrary()


new_parser_job = Job(name='parse_txt', job_format='code', job_type='data_processing', language='python', job=my_new_parser)

new_pipeline = Pipeline(PipelineStage(new))
new_pipeline.execute()
