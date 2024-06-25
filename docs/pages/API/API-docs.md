# API Endpoints Documentation

*Currently, the application provides the following main API endpoints:*


### 1. /api/answer 
**Description:**

This endpoint is used to request answers to user-provided questions.

**Request:**

**Method**: `POST`

**Headers**: Content-Type should be set to `application/json; charset=utf-8`

**Request Body**: JSON object with the following fields:
* `question` — The user's question.
* `history`  —  (Optional) Previous conversation history.
* `api_key`— Your API key.
* `embeddings_key`  —  Your embeddings key.
* `active_docs` — The location of active documentation.

Here is a JavaScript Fetch Request example:
```js
// answer (POST http://127.0.0.1:5000/api/answer)
fetch("http://127.0.0.1:5000/api/answer", {
      "method": "POST",
      "headers": {
            "Content-Type": "application/json; charset=utf-8"
      },
      "body": JSON.stringify({"question":"Hi","history":null,"api_key":"OPENAI_API_KEY","embeddings_key":"OPENAI_API_KEY",
      "active_docs": "javascript/.project/ES2015/openai_text-embedding-ada-002/"})
})
.then((res) => res.text())
.then(console.log.bind(console))
```

**Response**

In response, you will get a JSON document containing the `answer`, `query` and `result`:
```json
{
  "answer": "Hi there! How can I help you?\n",
  "query": "Hi",
  "result": "Hi there! How can I help you?\nSOURCES:"
}
```

### 2. /api/docs_check

**Description:**

This endpoint will make sure documentation is loaded on the server (just run it every time user is switching between libraries (documentations)).

**Request:**

**Method**: `POST`

**Headers**: Content-Type should be set to `application/json; charset=utf-8`

**Request Body**: JSON object with the field:
* `docs` — The location of the documentation:
```js
// docs_check (POST http://127.0.0.1:5000/api/docs_check)
fetch("http://127.0.0.1:5000/api/docs_check", {
      "method": "POST",
      "headers": {
            "Content-Type": "application/json; charset=utf-8"
      },
      "body": JSON.stringify({"docs":"javascript/.project/ES2015/openai_text-embedding-ada-002/"})
})
.then((res) => res.text())
.then(console.log.bind(console))
```

**Response:**

In response, you will get a JSON document like this one indicating whether the documentation exists or not:
```json
{
  "status": "exists"
}
```


### 3. /api/combine
**Description:**

This endpoint provides information about available vectors and their locations with a simple GET request.

**Request:**

**Method**: `GET`

**Response:**

Response will include:
* `date`
* `description`
* `docLink`
* `fullName`
* `language`
* `location` (local or docshub)
* `model`
* `name`
* `version`

Example of JSON in Docshub and local:

<img width="295" alt="image" src="https://user-images.githubusercontent.com/15183589/224714085-f09f51a4-7a9a-4efb-bd39-798029bb4273.png">

### 4. /api/upload
**Description:**

This endpoint is used to upload a file that needs to be trained, response is JSON with task ID, which can be used to check on task's progress.

**Request:**

**Method**: `POST`

**Request Body**: A multipart/form-data form with file upload and additional fields, including `user` and `name`.

HTML example:

```html
<form action="/api/upload" method="post" enctype="multipart/form-data" class="mt-2">
    <input type="file" name="file" class="py-4" id="file-upload">
    <input type="text" name="user" value="local" hidden>
    <input type="text" name="name" placeholder="Name:">
    
    <button type="submit" class="py-2 px-4 text-white bg-purple-30 rounded-md hover:bg-purple-30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-30">
        Upload
    </button>
</form>
```

**Response:**

JSON response with a status and a task ID that can be used to check the task's progress.


### 5. /api/task_status
**Description:**

This endpoint is used to get the status of a task (`task_id`) from `/api/upload`

**Request:**

**Method**: `GET`

**Query Parameter**: `task_id` (task ID to check)

**Sample JavaScript Fetch Request:**
```js
// Task status (Get http://127.0.0.1:5000/api/task_status)
fetch("http://localhost:5001/api/task_status?task_id=YOUR_TASK_ID", {
      "method": "GET",
      "headers": {
            "Content-Type": "application/json; charset=utf-8"
      },
})
.then((res) => res.text())
.then(console.log.bind(console))
```

**Response:**

There are two types of responses:

1. While the task is still running, the 'current' value will show progress from 0 to 100.
   ```json
   {
     "result": {
       "current": 1
     },
     "status": "PROGRESS"
   }
   ```

2. When task is completed:
   ```json
   {
     "result": {
       "directory": "temp",
       "filename": "install.rst",
       "formats": [
         ".rst",
         ".md",
         ".pdf"
       ],
       "name_job": "somename",
       "user": "local"
     },
     "status": "SUCCESS"
   }
   ```

### 6. /api/delete_old
**Description:**

This endpoint is used to delete old Vector Stores.

**Request:**

**Method**: `GET`

**Query Parameter**: `task_id`

**Sample JavaScript Fetch Request:**
```js
// delete_old (GET http://127.0.0.1:5000/api/delete_old)
fetch("http://localhost:5001/api/delete_old?task_id=YOUR_TASK_ID", {
      "method": "GET",
      "headers": {
            "Content-Type": "application/json; charset=utf-8"
      },
})
.then((res) => res.text())
.then(console.log.bind(console))

```
**Response:**

JSON response indicating the status of the operation:

```json
{ "status": "ok" }
```

### 7. /api/get_api_keys
**Description:**

The endpoint retrieves a list of API keys for the user.

**Request:**

**Method**: `GET`

**Sample JavaScript Fetch Request:**
```js
// get_api_keys (GET http://127.0.0.1:5000/api/get_api_keys)
fetch("http://localhost:5001/api/get_api_keys", {
      "method": "GET",
      "headers": {
            "Content-Type": "application/json; charset=utf-8"
      },
})
.then((res) => res.text())
.then(console.log.bind(console))

```
**Response:**

JSON response with a list of created API keys:

```json
[
      {
        "id": "string",
        "name": "string",
        "key": "string",
        "source": "string"
      },
      ...
    ]
```

### 8. /api/create_api_key

**Description:**

Create a new API key for the user.

**Request:**

**Method**: `POST`

**Headers**: Content-Type should be set to `application/json; charset=utf-8`

**Request Body**: JSON object with the following fields:
* `name` — A name for the API key.
* `source` — The source documents that will be used.
* `prompt_id` — The prompt ID.
* `chunks` — The number of chunks used to process an answer.

Here is a JavaScript Fetch Request example:
```js
// create_api_key (POST http://127.0.0.1:5000/api/create_api_key)
fetch("http://127.0.0.1:5000/api/create_api_key", {
      "method": "POST",
      "headers": {
            "Content-Type": "application/json; charset=utf-8"
      },
      "body": JSON.stringify({"name":"Example Key Name",
          "source":"Example Source",
          "prompt_id":"creative",
          "chunks":"2"})
})
.then((res) => res.json())
.then(console.log.bind(console))
```

**Response**

In response, you will get a JSON document containing the `id` and `key`:
```json
{
  "id": "string",
  "key": "string"
}
```

### 9. /api/delete_api_key

**Description:**

Delete an API key for the user.

**Request:**

**Method**: `POST`

**Headers**: Content-Type should be set to `application/json; charset=utf-8`

**Request Body**: JSON object with the field:
* `id` — The unique identifier of the API key to be deleted.

Here is a JavaScript Fetch Request example:
```js
// delete_api_key (POST http://127.0.0.1:5000/api/delete_api_key)
fetch("http://127.0.0.1:5000/api/delete_api_key", {
      "method": "POST",
      "headers": {
            "Content-Type": "application/json; charset=utf-8"
      },
      "body": JSON.stringify({"id":"API_KEY_ID"})
})
.then((res) => res.json())
.then(console.log.bind(console))
```

**Response:**

In response, you will get a JSON document indicating the status of the operation:
```json
{
  "status": "ok"
}
```