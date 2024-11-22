Engine
======

Engine
------
.. autoclass:: olive.engine.Engine
    :members: register, run

.. note::
    All parameters that of type :code:`...Config` or :code:`ConfigBase` class can be assigned dictionaries with keys corresponding
    to the fields of the class.

EngineConfig
^^^^^^^^^^^^
.. autopydantic_settings:: olive.engine.EngineConfig

**SearchStrategyConfig**

.. autopydantic_settings:: olive.strategy.search_strategy.SearchStrategyConfig


**SystemConfig**

.. autopydantic_settings:: olive.systems.system_config.SystemConfig
   :noindex:

**OliveEvaluatorConfig**

.. autopydantic_settings:: olive.evaluator.olive_evaluator.OliveEvaluatorConfig

SearchStrategy
^^^^^^^^^^^^^^
.. autoclass:: olive.strategy.search_strategy.SearchStrategy
